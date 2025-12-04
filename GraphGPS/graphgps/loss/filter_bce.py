import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch.nn as nn


@register_loss('filter_bce')
def filter_bce(pred, true):
    if 'filter_bce' in cfg.model.loss_fun:
        # print("compute bce")
        y=true
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
        # print(pred.shape)
        # print(y.shape)
        loss_mat = criterion(pred.double(), (y+1)/2)
        is_valid = y**2 > 0
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))  
        cls_loss = torch.sum(loss_mat)/torch.sum(is_valid)
        return cls_loss, pred
    
@register_loss('custom_mse')
def custom_mse(pred, true):
    mse_loss = torch.nn.MSELoss(reduction=cfg.model.size_average)
    if 'mse' in cfg.model.loss_fun:
        # print("compute mse")
        y=true
        true = true.float()
        return torch.sqrt(mse_loss(pred, true)), pred
