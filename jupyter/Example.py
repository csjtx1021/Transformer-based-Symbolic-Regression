import sys
 
sys.path.append(".")

from src.nesymres.architectures.model import Model
from src.nesymres.utils import load_metadata_hdf5
from src.nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
import torch
from sympy import lambdify
import json

## Load equation configuration and architecture configuration
import omegaconf




with open('./jupyter/100M/eq_setting.json', 'r') as json_file:
  eq_setting = json.load(json_file)

cfg = omegaconf.OmegaConf.load("./jupyter/100M/config.yaml")

## Set up BFGS load rom the hydra config yaml
bfgs = BFGSParams(
        activated= cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

params_fit = FitParams(word2id=eq_setting["word2id"],
                            id2word={int(k): v for k,v in eq_setting["id2word"].items()},
                            una_ops=eq_setting["una_ops"],
                            bin_ops=eq_setting["bin_ops"],
                            total_variables=list(eq_setting["total_variables"]),
                            total_coefficients=list(eq_setting["total_coefficients"]),
                            rewrite_functions=list(eq_setting["rewrite_functions"]),
                            bfgs=bfgs,
                            beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                            )

weights_path = "./jupyter/weights/100M.ckpt"

## Load architecture, set into eval mode, and pass the config parameters
model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
model.eval()
if torch.cuda.is_available():
    with torch.cuda.device(0):
        model.cuda()

fitfunc = partial(model.fitfunc,cfg_params=params_fit)



# Create points from an equation
number_of_points = 500
n_variables = 1

#To get best results make sure that your support inside the max and mix support
max_supp = cfg.dataset_train.fun_support["max"]
min_supp = cfg.dataset_train.fun_support["min"]
X = torch.rand(number_of_points,len(list(eq_setting["total_variables"])))*(max_supp-min_supp)+min_supp
X[:,n_variables:] = 0
# target_eq = "x_1*sin(x_1)" #Use x_1,x_2 and x_3 as independent variables
target_eq = "x_1*sin(x_1)" #Use x_1,x_2 and x_3 as independent variables
X_dict = {x:X[:,idx].cpu() for idx, x in enumerate(eq_setting["total_variables"])}
y = lambdify(",".join(eq_setting["total_variables"]), target_eq)(**X_dict)

print("X shape: ", X.shape)
print("y shape: ", y.shape)

import time

start_time = time.time()

prediction = fitfunc(X,y)

print(prediction)

print(time.time()-start_time)