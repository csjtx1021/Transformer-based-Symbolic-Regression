import time
from tqdm import tqdm
import numpy as np

import torch
import sympy as sp
import sympytorch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def opt_constant_batch(expressions_batch, data, device):
    X = data['x'].to(device)   # [#points, dim]
    Y = data['y'].to(device)   # [#points, #eqs]
    ##
    input_params = {}
    for dim_i in range(X.size(-1)):
        input_params['x_%s' % (dim_i+1)] = X[:, dim_i]
    ##
    mod = sympytorch.SymPyModule(expressions=expressions_batch).to(device)
    
    #for p in mod.parameters():
    #    print(p)
    
    optim = torch.optim.Adam(mod.parameters(), lr=1e-1)
    # optim = torch.optim.SGD(mod.parameters(), lr=1e-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.9, patience=20)

    num_epochs = 100000
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=1e-4)

    aa = time.time()

    for epoch in tqdm(range(num_epochs)):
        mod.train()

        optim.zero_grad()

        out = mod(**input_params)  # out has shape (200, 2)
        
        #print(out)

        # loss = torch.sum(torch.mean((out - y_truth) ** 2, dim=0))
        loss_each = torch.mean((out - Y) ** 2, dim=0)
        loss = torch.mean(loss_each)
        # loss = torch.sum(torch.mean(torch.abs(out - y_truth)/torch.abs(y_truth), dim=0))
        
        #print(loss)
        
        loss.backward()

        optim.step()  ## update paramters

        lr_scheduler.step(loss.item())

        early_stopping(loss.item(), mod)

        if early_stopping.early_stop or epoch >= num_epochs - 1:
            print('epoch #%s, loss = %.5f, time cost = %.2f' % (epoch, loss.item(), time.time() - aa))
            if early_stopping.early_stop:
                print("Early stopping")
            return [str(expr_i) for expr_i in mod.sympy()], loss_each.detach().cpu().numpy()


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #device = torch.device("cpu")

    print(device)

    def sample_uniform(min_=-5, max_=5):
        return np.random.rand() * (max_ - min_) + min_

    expressions = ["%.4f * cos(x_1) + %.4f + %.4f * cos(x_2) + %.4f " % (sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform()),
                   "%.4f * sin(x_1) + %.4f + %.4f * sin(x_2) + %.4f" % (sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform()),
                   "%.4f * x_1**2 + %.4f * x_1 + %.4f + %.4f * x_2**2 + %.4f * x_2 + %.4f" % (sample_uniform(), sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform(), sample_uniform()),
                   "%.4f * x_1 + %.4f + %.4f * x_2 + %.4f" % (sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform()),
                   ] * 100
    expressions = [sp.simplify(e) for e in expressions]

    support_max = 10.0
    support_min = -10.0

    x_ = torch.rand(500, 2) * (support_max - support_min) + support_min

    # ground truth
    y_truth = torch.cat([x_.cos().sum(-1).unsqueeze(-1),
                         -x_.sin().sum(-1).unsqueeze(-1),
                         (-0.5 * x_ ** 2 - x_).sum(-1).unsqueeze(-1),
                         (-0.5 * x_ ** 2 - x_).sum(-1).unsqueeze(-1),
                         ] * 100, dim=-1)

    data = {
        'x': x_,  # [#points, dim]
        'y': y_truth  # [#points, #eqs]
    }

    opt_sympy = opt_constant_batch(expressions, data, device)

    print(opt_sympy)
