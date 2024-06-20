import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from .set_encoder import SetEncoder
from .beam_search import BeamHypotheses
import numpy as np
from tqdm import tqdm 
from ..dataset.generator import Generator, InvalidPrefixExpression
from itertools import chain
from sympy import lambdify 
from . import bfgs
from . import data

import sympy as sp

from ..dataset.sympy_utils import add_multiplicative_constants, add_additive_constants

from . import opt_constant_fast

import time



class Model(pl.LightningModule):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.enc = SetEncoder(cfg)
        self.trg_pad_idx = cfg.trg_pad_idx
        self.tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
        if cfg.sinuisodal_embeddings:
            self.create_sinusoidal_embeddings(
                cfg.length_eq, cfg.dim_hidden, out=self.pos_embedding.weight
            )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_hidden,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dec_pf_dim,
            dropout=cfg.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.dec_layers)
        self.fc_out = nn.Linear(cfg.dim_hidden, cfg.output_dim)
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.eq = None
    

    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask

    def forward(self,batch):
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, : (size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)
        trg = batch[1].long()
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])
        src_mask = None
        encoder_input = torch.cat((src_x, src_y), dim=-1)
        enc_src = self.enc(encoder_input) 
        assert not torch.isnan(enc_src).any()
        pos = self.pos_embedding(
            torch.arange(0, batch[1].shape[1] - 1)
            .unsqueeze(0)
            .repeat(batch[1].shape[0], 1)
            .type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        trg_ = self.dropout(te + pos)
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_src.permute(1, 0, 2),
            trg_mask2.bool(),
            tgt_key_padding_mask=trg_mask1.bool(),
        ) 
        output = self.fc_out(output)
        return output, trg

    def compute_loss(self,output, trg):
        output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return loss

    def training_step(self, batch, _):
        output, trg = self.forward(batch)
        loss = self.compute_loss(output,trg)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        output, trg = self.forward(batch)
        loss = self.compute_loss(output,trg)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def fitfunc(self, X,y, cfg_params=None):
        """Same API as fit functions in sklearn: 
            X [Number_of_points, Number_of_features], 
            Y [Number_of_points]
        """
        X = X
        y = y[:,None]
        
        X = torch.tensor(X,device=self.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1],self.cfg.dim_input-X.shape[2]-1, device=self.device)
            X = torch.cat((X,pad),dim=2)
        y = torch.tensor(y,device=self.device).unsqueeze(0)
        with torch.no_grad():

            encoder_input = torch.cat((X, y), dim=2) #.permute(0, 2, 1)
            # if self.device.type == "cuda":
            #     encoder_input = encoder_input.cuda()
            enc_src = self.enc(encoder_input)
            src_enc = enc_src
            shape_enc_src = (cfg_params.beam_size,) + src_enc.shape[1:]
            enc_src = src_enc.unsqueeze(1).expand((1, cfg_params.beam_size) + src_enc.shape[1:]).contiguous().view(shape_enc_src)
            print(
                "Memory footprint of the encoder: {}GB \n".format(
                    enc_src.element_size() * enc_src.nelement() / 10 ** (9)
                )
            )
            assert enc_src.size(0) == cfg_params.beam_size
            generated = torch.zeros(
                [cfg_params.beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )
            generated[:, 0] = 1
            # trg_indexes = [[1] for i in range(bs*self.beam_size)]
            cache = {"slen": 0}
            # generated = torch.tensor(trg_indexes,device=self.device,dtype=torch.long)
            generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)
            done = False 
            # Beam Scores
            beam_scores = torch.zeros(cfg_params.beam_size, device=self.device, dtype=torch.long)
            beam_scores[1:] = -1e9
            #beam_scores = beam_scores.view(-1)

            cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
            while cur_len < self.cfg.length_eq:
                # breakpoint()
                generated_mask1, generated_mask2 = self.make_trg_mask(
                    generated[:, :cur_len]
                )

                # dec_args = (generated, enc_src, generated_mask, src_mask)

                pos = self.pos_embedding(
                    torch.arange(0, cur_len)  #### attention here
                    .unsqueeze(0)
                    .repeat(generated.shape[0], 1)
                    .type_as(generated)
                )
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = self.dropout(te + pos)

                output = self.decoder_transfomer(
                    trg_.permute(1, 0, 2),
                    enc_src.permute(1, 0, 2),
                    generated_mask2.float(),
                    tgt_key_padding_mask=generated_mask1.bool(),
                )
                output = self.fc_out(output)
                output = output.permute(1, 0, 2).contiguous()
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(
                    1
                ) 
                
                assert output[:, -1:, :].shape == (cfg_params.beam_size,1,self.cfg.length_eq,)

                n_words = scores.shape[-1]
                # select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(
                    scores
                )  # (bs * beam_size, n_words)
                _scores = _scores.view(cfg_params.beam_size * n_words)  # (bs, beam_size * n_words)

                next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=0, largest=True, sorted=True)
                assert len(next_scores) == len(next_words) == 2 * cfg_params.beam_size
                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words, next_scores):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if (
                        word_id == cfg_params.word2id["F"]
                        or cur_len + 1 == self.cfg.length_eq
                    ):
                        generated_hyps.add(
                            generated[
                                 beam_id,
                                :cur_len,
                            ]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == cfg_params.beam_size:
                        break

                # update next beam content
                assert (
                    len(next_sent_beam) == 0
                    if cur_len + 1 == self.cfg.length_eq
                    else cfg_params.beam_size
                )
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.trg_pad_idx, 0)
                    ] * cfg_params.beam_size  # pad the batch


                #next_batch_beam.extend(next_sent_beam)
                assert len(next_sent_beam) == cfg_params.beam_size

                beam_scores = torch.tensor(
                    [x[0] for x in next_sent_beam], device=self.device
                )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                beam_words = torch.tensor(
                    [x[1] for x in next_sent_beam], device=self.device
                )  # generated.new([x[1] for x in next_batch_beam])
                beam_idx = torch.tensor(
                    [x[2] for x in next_sent_beam], device=self.device
                )
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

                # update current length
                cur_len = cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )


            #perc = 0
            #cnt = 0
            #gts = []
            best_preds = []
            best_preds_bfgs = []
            #best_L = []
            best_L_bfgs = []

            #flag = 0
            L_bfgs = []
            P_bfgs = []
            #counter = 1

            #fun_args = ",".join(chain(cfg_params.total_variables,"constant"))
            cfg_params.id2word[3] = "constant"
            for __, ww in sorted(
                generated_hyps.hyp, key=lambda x: x[0], reverse=True
            ):
                try:
                    pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(
                        ww, X, y, cfg_params
                    )
                except InvalidPrefixExpression:
                    continue
                #L_bfgs = loss_bfgs
                P_bfgs.append(str(pred_w_c))
                L_bfgs.append(loss_bfgs)

            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = None
            else:
                best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                best_L_bfgs.append(np.nanmin(L_bfgs))

            output = {'all_bfgs_preds':P_bfgs, 'all_bfgs_loss':L_bfgs, 'best_bfgs_preds':best_preds_bfgs, 'best_bfgs_loss':best_L_bfgs}
            self.eq = output['best_bfgs_preds']
            return output


    
    def sample_uniform(self, min_=-5, max_=5):
        return np.random.rand() * (max_ - min_) + min_

    def fitfunc_batch(self, X,y, cfg_params=None, fast_flag=True):
        """Same API as fit functions in sklearn: 

            X [Number_of_points, Number_of_features], 
            Y [batch, Number_of_points]
        """
        X = X.unsqueeze(0).repeat(y.shape[0],1,1)  # [batch, Number_of_points, Number_of_features]
        y = y[:,:,None]                            # [batch, Number_of_points, 1]
        
        batch_size = X.shape[0]
        
        print('batch_size = %s'%batch_size)
        
        #X = torch.tensor(X,device=self.device).unsqueeze(0)
        X = torch.tensor(X,device=self.device)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1],self.cfg.dim_input-X.shape[2]-1, device=self.device)
            X = torch.cat((X,pad),dim=2)
        #y = torch.tensor(y,device=self.device).unsqueeze(0)
        y = torch.tensor(y,device=self.device)
        with torch.no_grad():
        
            print("\r\n**encoding and decoding ... \r\n")
            start_forward = time.time()
            """
            encoder_input = torch.cat((X, y), dim=2) #.permute(0, 2, 1)
            # if self.device.type == "cuda":
            #     encoder_input = encoder_input.cuda()
            enc_src = self.enc(encoder_input)
            src_enc = enc_src
            #print(encoder_input.shape, src_enc.shape)
            
            generated_hyps_batch = []
            
            for batch_idx in tqdm(range(batch_size)):
                
                #print('decoding #%s/%s'%(batch_idx, batch_size))
                
                shape_enc_src = (cfg_params.beam_size,) + src_enc[batch_idx].shape
                #print(shape_enc_src)
                #exit(1)
                enc_src_i = src_enc[batch_idx].unsqueeze(0).unsqueeze(1).expand((1, cfg_params.beam_size) + src_enc[batch_idx].shape).contiguous().view(shape_enc_src)
                
                
                #print(
                #    "Memory footprint of the encoder: {}GB \n".format(
                #        enc_src_i.element_size() * enc_src_i.nelement() / 10 ** (9)
                #    )
                #)
                assert enc_src_i.size(0) == cfg_params.beam_size
                generated = torch.zeros(
                    [cfg_params.beam_size, self.cfg.length_eq],
                    dtype=torch.long,
                    device=self.device,
                )
                generated[:, 0] = 1
                # trg_indexes = [[1] for i in range(bs*self.beam_size)]
                cache = {"slen": 0}
                # generated = torch.tensor(trg_indexes,device=self.device,dtype=torch.long)
                generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)
                done = False 
                # Beam Scores
                beam_scores = torch.zeros(cfg_params.beam_size, device=self.device, dtype=torch.long)
                beam_scores[1:] = -1e9
                #beam_scores = beam_scores.view(-1)
    
                cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
                while cur_len < self.cfg.length_eq:
                    # breakpoint()
                    generated_mask1, generated_mask2 = self.make_trg_mask(
                        generated[:, :cur_len]
                    )
    
                    # dec_args = (generated, enc_src, generated_mask, src_mask)
    
                    pos = self.pos_embedding(
                        torch.arange(0, cur_len)  #### attention here
                        .unsqueeze(0)
                        .repeat(generated.shape[0], 1)
                        .type_as(generated)
                    )
                    te = self.tok_embedding(generated[:, :cur_len])
                    trg_ = self.dropout(te + pos)
    
                    output = self.decoder_transfomer(
                        trg_.permute(1, 0, 2),
                        enc_src_i.permute(1, 0, 2),
                        generated_mask2.float(),
                        tgt_key_padding_mask=generated_mask1.bool(),
                    )
                    output = self.fc_out(output)
                    output = output.permute(1, 0, 2).contiguous()
                    scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(
                        1
                    ) # (cfg_params.beam_size*batch_size,self.cfg.length_eq,)
                    
                    assert output[:, -1:, :].shape == (cfg_params.beam_size,1,self.cfg.length_eq,)
    
                    n_words = scores.shape[-1]  
                    # select next words with scores
                    _scores = scores + beam_scores[:, None].expand_as(
                        scores
                    )  # (bs * beam_size, n_words)
                    _scores = _scores.view(cfg_params.beam_size * n_words)  # (bs, beam_size * n_words)
    
                    next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=0, largest=True, sorted=True)
                    assert len(next_scores) == len(next_words) == 2 * cfg_params.beam_size
                    done = done or generated_hyps.is_done(next_scores.max().item())
                    next_sent_beam = []
    
                    # next words for this sentence
                    for idx, value in zip(next_words, next_scores):
    
                        # get beam and word IDs
                        beam_id = idx // n_words
                        word_id = idx % n_words
    
                        # end of sentence, or next word
                        if (
                            word_id == cfg_params.word2id["F"]
                            or cur_len + 1 == self.cfg.length_eq
                        ):
                            generated_hyps.add(
                                generated[
                                     beam_id,
                                    :cur_len,
                                ]
                                .clone()
                                .cpu(),
                                value.item(),
                            )
                        else:
                            next_sent_beam.append(
                                (value, word_id, beam_id)
                            )
    
                        # the beam for next step is full
                        if len(next_sent_beam) == cfg_params.beam_size:
                            break
    
                    # update next beam content
                    assert (
                        len(next_sent_beam) == 0
                        if cur_len + 1 == self.cfg.length_eq
                        else cfg_params.beam_size
                    )
                    if len(next_sent_beam) == 0:
                        next_sent_beam = [
                            (0, self.trg_pad_idx, 0)
                        ] * cfg_params.beam_size  # pad the batch
    
    
                    #next_batch_beam.extend(next_sent_beam)
                    assert len(next_sent_beam) == cfg_params.beam_size
    
                    beam_scores = torch.tensor(
                        [x[0] for x in next_sent_beam], device=self.device
                    )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                    beam_words = torch.tensor(
                        [x[1] for x in next_sent_beam], device=self.device
                    )  # generated.new([x[1] for x in next_batch_beam])
                    beam_idx = torch.tensor(
                        [x[2] for x in next_sent_beam], device=self.device
                    )
                    generated = generated[beam_idx, :]
                    generated[:, cur_len] = beam_words
                    for k in cache.keys():
                        if k != "slen":
                            cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])
    
                    # update current length
                    cur_len = cur_len + torch.tensor(
                        1, device=self.device, dtype=torch.int64
                    )
                    
                ### saved
                generated_hyps_batch.append(generated_hyps)
            """    
            encoder_input = torch.cat((X, y), dim=2) #.permute(0, 2, 1)
            # if self.device.type == "cuda":
            #     encoder_input = encoder_input.cuda()
            enc_src = self.enc(encoder_input)
            src_enc = enc_src
            #print(encoder_input.shape, src_enc.shape)
            
            generated_hyps_batch = []
            
            #for batch_idx in tqdm(range(batch_size)):
            if True:    
                #print('decoding #%s/%s'%(batch_idx, batch_size))
                
                shape_enc_src = (batch_size*cfg_params.beam_size,) + src_enc.shape[1:]
                #print(shape_enc_src)
                #exit(1)
                # [batch_size, beam_size, dim]
                enc_src = src_enc.unsqueeze(1).expand((batch_size, cfg_params.beam_size) + src_enc.shape[1:]).contiguous().view(shape_enc_src)
                
                #print(
                #    "Memory footprint of the encoder: {}GB \n".format(
                #        enc_src_i.element_size() * enc_src_i.nelement() / 10 ** (9)
                #    )
                #)
                # assert enc_src_i.size(0) == cfg_params.beam_size
                generated = torch.zeros(
                    [batch_size, cfg_params.beam_size, self.cfg.length_eq],
                    dtype=torch.long,
                    device=self.device,
                )
                generated[:, :, 0] = 1
                # trg_indexes = [[1] for i in range(bs*self.beam_size)]
                cache = {"slen": 0}
                # generated = torch.tensor(trg_indexes,device=self.device,dtype=torch.long)
                
                generated_hyps_batch = []
                for _ in range(batch_size):
                    generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)
                    generated_hyps_batch.append(generated_hyps)
                    
                done = False 
                # Beam Scores
                beam_scores = torch.zeros((batch_size, cfg_params.beam_size), device=self.device, dtype=torch.long)
                beam_scores[:, 1:] = -1e9
                #beam_scores = beam_scores.view(-1)
    
                cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
                while cur_len < self.cfg.length_eq:
                    # breakpoint()
                    generated_mask1, generated_mask2 = self.make_trg_mask(
                        generated.view(-1, self.cfg.length_eq)[:, :cur_len]
                    )
    
                    # dec_args = (generated, enc_src, generated_mask, src_mask)
    
                    pos = self.pos_embedding(
                        torch.arange(0, cur_len)  #### attention here
                        .unsqueeze(0)
                        .repeat(batch_size * cfg_params.beam_size, 1)
                        .type_as(generated)
                    )
                    te = self.tok_embedding(generated.view(-1, self.cfg.length_eq)[:, :cur_len])
                    trg_ = self.dropout(te + pos)
    
                    output = self.decoder_transfomer(
                        trg_.permute(1, 0, 2),
                        enc_src.permute(1, 0, 2),
                        generated_mask2.float(),
                        tgt_key_padding_mask=generated_mask1.bool(),
                    )
                    output = self.fc_out(output)
                    output = output.permute(1, 0, 2).contiguous()
                    scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(
                        1
                    ) # (cfg_params.beam_size*batch_size,self.cfg.length_eq,)
                    
                    # assert output[:, -1:, :].shape == (cfg_params.beam_size,1,self.cfg.length_eq,)
                    # [batch_size, cfg_params.beam_size, self.cfg.length_eq]
                    scores = scores.view(batch_size, cfg_params.beam_size, self.cfg.length_eq)
                    
                    n_words = scores.shape[-1]  
                    # select next words with scores
                    # [batch_size, cfg_params.beam_size, self.cfg.length_eq]
                    _scores = scores + beam_scores[:, :, None].expand_as(
                        scores
                    )  
                    # (batch_size, beam_size*n_words)
                    _scores = _scores.view(batch_size, cfg_params.beam_size * n_words)  # (bs, beam_size * n_words)
                    # (batch_size, 2 * cfg_params.beam_size)
                    next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=1, largest=True, sorted=True)
                    assert next_scores.size(-1) == next_words.size(-1) == 2 * cfg_params.beam_size
                    
                    for batch_idx in range(batch_size):
                    
                        done = done or generated_hyps_batch[batch_idx].is_done(next_scores[batch_idx].max().item())
                        next_sent_beam = []
                        # next words for this sentence
                        for idx, value in zip(next_words[batch_idx], next_scores[batch_idx]):
        
                            # get beam and word IDs
                            beam_id = idx // n_words
                            word_id = idx % n_words
        
                            # end of sentence, or next word
                            if (
                                word_id == cfg_params.word2id["F"]
                                or cur_len + 1 == self.cfg.length_eq
                            ):
                                generated_hyps_batch[batch_idx].add(
                                    generated[batch_idx][
                                         beam_id,
                                        :cur_len,
                                    ]
                                    .clone()
                                    .cpu(),
                                    value.item(),
                                )
                            else:
                                next_sent_beam.append(
                                    (value, word_id, beam_id)
                                )
        
                            # the beam for next step is full
                            if len(next_sent_beam) == cfg_params.beam_size:
                                break

                        # update next beam content
                        assert (
                            len(next_sent_beam) == 0
                            if cur_len + 1 == self.cfg.length_eq
                            else cfg_params.beam_size
                        )
                        if len(next_sent_beam) == 0:
                            next_sent_beam = [
                                (0, self.trg_pad_idx, 0)
                            ] * cfg_params.beam_size  # pad the batch
    
    
                        #next_batch_beam.extend(next_sent_beam)
                        assert len(next_sent_beam) == cfg_params.beam_size
        
                        beam_scores[batch_idx] = torch.tensor(
                            [x[0] for x in next_sent_beam], device=self.device
                        )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                        beam_words = torch.tensor(
                            [x[1] for x in next_sent_beam], device=self.device
                        )  # generated.new([x[1] for x in next_batch_beam])
                        beam_idx = torch.tensor(
                            [x[2] for x in next_sent_beam], device=self.device
                        )
                        
                        
                        generated[batch_idx,:,:] = generated[batch_idx][beam_idx, :]
                        generated[batch_idx, :, cur_len] = beam_words
                        for k in cache.keys():
                            if k != "slen":
                                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])
    
                    # update current length
                    cur_len = cur_len + torch.tensor(
                        1, device=self.device, dtype=torch.int64
                    )
                    
        if fast_flag is False:
                #perc = 0
                #cnt = 0
                #gts = []
                best_preds = []
                best_preds_bfgs = []
                #best_L = []
                best_L_bfgs = []
    
                #flag = 0
                L_bfgs = []
                P_bfgs = []
                #counter = 1
    
                #fun_args = ",".join(chain(cfg_params.total_variables,"constant"))
                cfg_params.id2word[3] = "constant"
                
                for batch_idx in range(len(generated_hyps_batch)):
                    generated_hyps = generated_hyps_batch[batch_idx]
                    count = -1
                    for __, ww in sorted(
                        generated_hyps.hyp, key=lambda x: x[0], reverse=True
                    ): 
                        try:
                            count += 1
                            print('batch #%s, beam %s'%(batch_idx, count))
                            pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(
                                ww, X[batch_idx:batch_idx+1,:,:], y[batch_idx:batch_idx+1,:,:], cfg_params
                            )
                        except InvalidPrefixExpression:
                            continue
                        #L_bfgs = loss_bfgs
                        P_bfgs.append(str(pred_w_c))
                        L_bfgs.append(loss_bfgs)
        
                    if all(np.isnan(np.array(L_bfgs))):
                        print("Warning all nans")
                        L_bfgs = float("nan")
                        best_L_bfgs = None
                    else:
                        best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                        best_L_bfgs.append(np.nanmin(L_bfgs))
    
                output = {'all_bfgs_preds':P_bfgs, 'all_bfgs_loss':L_bfgs, 'best_bfgs_preds':best_preds_bfgs, 'best_bfgs_loss':best_L_bfgs}
                self.eq = output['best_bfgs_preds']
                return output
        else:
                #perc = 0
                #cnt = 0
                #gts = []
                best_preds = []
                best_preds_bfgs = []
                #best_L = []
                best_L_bfgs = []
    
                #flag = 0
                L_bfgs = []
                P_bfgs = []
                #counter = 1
                
                expressions = []
    
                #fun_args = ",".join(chain(cfg_params.total_variables,"constant"))
                cfg_params.id2word[3] = "constant"
                
                for batch_idx in range(len(generated_hyps_batch)):
                    generated_hyps = generated_hyps_batch[batch_idx]
                    for __, ww in sorted(
                        generated_hyps.hyp, key=lambda x: x[0], reverse=True
                    ):  
                        #print(ww)
                        ww1 = ww[1:].tolist()
                        #pred_str = [x if x<14 else x+1 for x in pred_str]
                        raw = data.de_tokenize(ww1, cfg_params.id2word)
                        #print(raw)
                        
                        #raw = ['mul', 'x_1', 'add', 'mul', '-1', 'mul', 'x_2', 'pow', 'x_1', '3', 'mul', 'constant', 'mul', 'pow', 'x_1', '-1', 'x_1', 'add', 'x_1', 'constant']
                        
                        try:
                            candidate_expr = Generator.prefix_to_infix(raw, 
                                                            coefficients=["constant"], 
                                                            variables=cfg_params.total_variables)
                            candidate_expr = candidate_expr.format(constant="constant")
                            
                            if 'constant' not in candidate_expr:
                                candidate_expr = candidate_expr + " + constant"
                                
                        except InvalidPrefixExpression:
                            candidate_expr = "constant"
                        
                        
                        expr_str = candidate_expr
                        for _ in range(candidate_expr.count("constant")):
                            expr_str = expr_str.replace("constant", '%.4f'%self.sample_uniform(),1)
                        #expr_str = '(%s) * %.4f + %.4f'%(expr_str, self.sample_uniform(), self.sample_uniform())
                        #expr_str = '%s + %.4f'%(expr_str, self.sample_uniform())
                        #print(candidate_expr, expr_str)
                        expressions.append(sp.simplify(expr_str))
                        
                        #print(sp.sympify(candidate_expr))
                
                print("\r\n**forward and beam search ok!, time cost = %.2f\r\n"%(time.time()-start_forward))   
                print(expressions)
                
                data_feed = {
                    'x': X[0].detach(),  # [#points, dim]
                    'y': y[:,:,0].unsqueeze(1).repeat(1, cfg_params.beam_size, 1).view(cfg_params.beam_size*batch_size, -1).permute(1,0).detach()  # [#points, #eqs]
                    
                }
                #print(data_feed['x'].size(), data_feed['y'].size())
    
                opt_expr_str, loss_each = opt_constant_fast.opt_constant_batch(expressions, data_feed, self.device)
                
                #print(loss_each)
                #print(np.array(opt_expr_str))
                #print(opt_sympy)
                output = {'all_preds': np.array(opt_expr_str).reshape(batch_size,-1),
                          'all_loss': loss_each.reshape(batch_size,-1),
                          'best_preds': [np.array(opt_expr_str).reshape(batch_size,-1)[idx][np.nanargmin(loss_each.reshape(batch_size,-1)[idx])] for idx in range(batch_size)],
                          'best_loss': [np.nanmin(loss_each.reshape(batch_size,-1)[idx]) for idx in range(batch_size)]}
                
                return output

    def get_equation(self,):
        return self.eq


if __name__ == "__main__":
        model = SetTransformer(n_l_enc=2,src_pad_idx=0,trg_pad_idx=0,dim_input=6,output_dim=20,dim_hidden=40,dec_layers=1,num_heads=8,dec_pf_dim=40,dec_dropout=0,length_eq=30,lr=
            0.001,num_inds=20,ln=True,num_features=10,is_sin_emb=False, bit32=True,norm=False,activation='linear',linear=False,mean=torch.Tensor([1.]),std=torch.Tensor([1.]),input_normalization=False)
        src_x = torch.rand([2,5,20])
        src_y = torch.sin(torch.norm(src_x, dim=1)).unsqueeze(1)
        inp_1 = torch.cat([src_x,src_y], dim=1)
        inp_2 = torch.randint(0,13,[2,10])
        batch = (inp_1,inp_2)
        print(model)