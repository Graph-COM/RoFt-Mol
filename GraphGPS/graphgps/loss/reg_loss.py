# import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch.nn as nn
import torch

def l2_sp(target_model, source_model):
    # print("compute l2 sp")
    source_weight = {}
    output = 0.0
    for name, param in source_model.named_parameters():
        source_weight[name] = param.detach()

    for name, param in target_model.named_parameters():
        output += 0.5 * torch.norm(param - source_weight[name]) ** 2
    return output
    
def feature_map(target_model, source_model, batch):
    # print("compute feature map")
    output = 0.0
    batch_ft = batch.clone()
    source_model(batch)
    target_model(batch_ft)
    for fm_src, fm_tgt in zip(batch.x, batch_ft.x):
        output += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
    return output

def bss(feature):
    # print('bss')
    result=0
    u, s, v = torch.svd(feature.t())
    num = s.size(0)
    for i in range(1):
        result += torch.pow(s[num-1-i], 2)
    return result

