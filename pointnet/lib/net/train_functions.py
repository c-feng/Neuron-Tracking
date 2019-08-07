import torch
import torch.nn as nn
import utils.loss_utils as loss_utils
# from utils.loss_utils import discriminative_loss
from lib.config import cfg
from collections import namedtuple
import pdb

def model_fn_neural_ins_decorator():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', "disp_dict"])

    def model_fn(model, data):
        point_sets, ins = data

        point_sets = torch.from_numpy(point_sets).cuda().float()
        ins = torch.from_numpy(ins).cuda().float()

        embeded_feature = model(point_sets)
        
        embeded_feature = embeded_feature.permute(0, 2, 1)
        loss, lv, ld, lr = loss_utils.discriminative_loss(embeded_feature, ins,
                                                          cfg.LOSS.DELTA_V, cfg.LOSS.DELTA_D,
                                                          cfg.LOSS.PARAM_VAR, cfg.LOSS.PARAM_DIST, cfg.LOSS.PARAM_REG)

        # record data
        tb_dict = {}
        disp_dict = {}
        tb_dict.update({"L_variace": lv.item(), "L_distance": ld.item(),
                        "L_reg": lr.item()})
        
        disp_dict.update({"Loss": loss.item()})

        return ModelReturn(loss, tb_dict, disp_dict)
    
    return model_fn

def model_fn_neural_ins_sim_decoratot():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', "disp_dict"])

    def model_fn(model, data):
        point_sets, ins = data

        point_sets = torch.from_numpy(point_sets).cuda().float()
        ins = torch.from_numpy(ins).cuda().float()

        # sim_feat: (B, npoints, nfeature)
        # conf_map: (B, npoints)
        sim_feat, conf_map = model(point_sets)
        sim_feat = sim_feat.permute(0, 2, 1)  # (B, npoint, nfeature)

        






        return ModelReturn(loss, tb_dict, disp_dict)
    
    return model_fn
