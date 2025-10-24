import torch
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import math

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import itertools

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error
from ftlib.reg_methods.bss import BatchSpectralShrinkage
from ftlib.reg_methods.delta import IntermediateLayerGetter, L2Regularization, get_attribute, BehavioralRegularization, AttentionBehavioralRegularization
from ftlib.reg_methods.delta import SPRegularization, FrobeniusRegularization
from ftlib.reg_methods.gtot_tuning import GTOTRegularization
from ftlib.reg_methods.meter import AverageMeter, ProgressMeter
from ftlib.reg_methods.eval import Meter


from splitters import scaffold_split, random_split, random_scaffold_split, size_split
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter


criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion_reg = nn.MSELoss()

def train(args, epoch, model, device, loader, optimizer, weights_regularization, backbone_regularization, 
                head_regularization, target_getter,
                source_getter, bss_regularization):
    model.train()
    epoch_iter = tqdm(loader, desc="Iteration")
    loss_epoch = []
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)

        intermediate_output_s, output_s = source_getter(batch)  # batch.batch is a column vector which maps each node to its respective graph in the batch
        intermediate_output_t, output_t = target_getter(batch)
        fea = global_mean_pool(output_t[1], batch.batch)
        pred, node_representation = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        
        #Loss matrix
        if args.regression:
            loss_mat = criterion_reg(pred.double(), y)
            loss_mat = torch.sqrt(loss_mat)
            is_valid = torch.ones_like(y).bool()
        else:
            loss_mat = criterion(pred.double(), (y+1)/2)
            is_valid = y**2 > 0
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))  
        optimizer.zero_grad()
        cls_loss = torch.sum(loss_mat)/torch.sum(is_valid)

        loss_reg_head = head_regularization()
        loss_reg_backbone = 0.0
        print_str = ''
        loss = torch.tensor([0.0], device=device)
        loss_bss = 0.0
        loss_weights = torch.tensor([0.0]).to(cls_loss.device)
        if args.regularization_type == 'feature_map':
            loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t)
        elif args.regularization_type == 'attention_feature_map':
            loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t)
        elif args.regularization_type == 'l2_sp':
            loss_reg_backbone = backbone_regularization()
        elif args.regularization_type == 'bss':
            fea = fea if fea is not None else global_mean_pool(node_representation, batch.batch)
            loss_bss = bss_regularization(fea)  # if fea is not None else 0.0
        elif args.regularization_type == 'none':
            loss_reg_backbone = 0.0
            # loss_reg_head = 0.0
            loss_bss = 0.0
        elif args.regularization_type in ['gtot_feature_map',]:
            if args.trade_off_backbone > 0.0:
                loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t, batch)
            if False and 'best_' in args.tag:
                loss_weights = weights_regularization()
                print_str += f'loss_weights:{loss_weights:.5f}'
        else:
            loss_reg_backbone = backbone_regularization()

        loss = loss + cls_loss + args.trade_off_backbone * loss_reg_backbone + args.trade_off_head * loss_reg_head + args.trade_off_bss * loss_bss
        loss = loss + 0.1 * loss_weights
        # if torch.isnan(cls_loss):  # or torch.isnan(loss_reg_backbone):
        #     print(pred, loss_reg_backbone)
        #     raise
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        optimizer.step()
        epoch_iter.set_description(f"Epoch: {epoch} tloss: {cls_loss:.4f}")

        # print(f'{"vanilla model || " if fea is None and args.norm_type == "none" else ""} '
        # f'cls_loss:{cls_loss:.5f}, loss_reg_backbone: {args.trade_off_backbone * loss_reg_backbone:.5f} loss_reg_head:'
        # f' {args.trade_off_head * loss_reg_head:.5f} bss_los: {args.trade_off_bss * loss_bss:.5f} ' + print_str)


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    if args.regression:
        roc_list.append(math.sqrt(mean_squared_error(y_true, y_scores)))
    else:
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def main():
    torch.set_num_threads(10)
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=5,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'lipo', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'Mole-BERT', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = 'debug', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--tune_option', type=str, default = "linear_layer", help='number of workers for dataset loading')
    parser.add_argument('--regression', type=bool, default = False, help='whether regression task')
    parser.add_argument('--fewshot', type=bool, default = False, help='whether few shot')
    parser.add_argument('--fewshot_num', type=int, default = 50, help='few shot number for the labeled data')
    # regularization based model parameter
    parser.add_argument('--regularization_type', type=str, # choices=['l2_sp', 'feature_map', 'attention_feature_map',"none"],
                        default='l2_sp', help='fine tune regularization.')
    parser.add_argument('--finetune_type', type=str, default='l2_sp', help='fine tune regularization.')  # choices=['delta', 'bitune', 'co_tune','l2_sp','none','bss'],
    parser.add_argument('--norm_type', type=str, default='none', help='fine tune regularization.')
    parser.add_argument('--trade_off_backbone', default=1, type=float, help='trade-off for backbone regularization')
    parser.add_argument('--trade_off_head', default=1, type=float, help='trade-off for head regularization')
    ## bss
    parser.add_argument('--trade_off_bss', default=1, type=float, help='trade-off for bss regularization')
    parser.add_argument('-k', '--k', default=1, type=int, metavar='N', help='hyper-parameter for BSS loss')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    score_list = []
    allseed = [0, 1, 2, 3, 4]
    for seed in allseed:
        args.runseed = seed
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)

        #Bunch of classification tasks
        if args.dataset == "tox21":
            num_tasks = 12
        elif args.dataset == "hiv":
            num_tasks = 1
        elif args.dataset == "pcba":
            num_tasks = 128
        elif args.dataset == "muv":
            num_tasks = 17
        elif args.dataset == "bace":
            num_tasks = 1
        elif args.dataset == "bbbp":
            num_tasks = 1
        elif args.dataset == "toxcast":
            num_tasks = 617
        elif args.dataset == "sider":
            num_tasks = 27
        elif args.dataset == "clintox":
            num_tasks = 2
        elif args.dataset in ['esol', 'lipo', 'freesolv', 'malaria', 'cep', 'mpbp']:
            num_tasks = 1
            args.regression = True
        else:
            raise ValueError("Invalid dataset name.")
        
        if args.dataset == 'mpbp':
            global criterion_reg 
            criterion_reg = nn.L1Loss()

        #set up dataset
        dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)
        print(dataset)
        
        if args.split == "scaffold":
            smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, args.fewshot, args.fewshot_num, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("scaffold")
        elif args.split == "size":
            smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = size_split(dataset, args.fewshot, args.fewshot_num, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, args.fewshot, args.fewshot_num, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        print('++++++++++', train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        #set up model
        model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        source_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        if not args.input_model_file == "None":
            print('Not from scratch')
            model.from_pretrained('model_gin/{}.pth'.format(args.input_model_file))
            source_model.from_pretrained('model_gin/{}.pth'.format(args.input_model_file))
        
        for param in source_model.parameters():
            param.requires_grad = False
            source_model.eval()
        
        model.to(device)
        source_model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        return_layers = ['gnn.gnns.4.mlp.2'] # for GIN
        source_getter = IntermediateLayerGetter(source_model, return_layers=return_layers)
        target_getter = IntermediateLayerGetter(model, return_layers=return_layers)

         # get regularization for finetune
        weights_regularization = FrobeniusRegularization(source_model, model)
        backbone_regularization = lambda x: x
        bss_regularization = lambda x: x

        if args.regularization_type in ['gtot_feature_map']:
            ''' the proposed method GTOT-tuning'''
            backbone_regularization = GTOTRegularization(order=args.gtot_order, args=args)
        #------------------------------ baselines --------------------------------------------
        elif args.regularization_type == 'l2_sp':
            backbone_regularization = SPRegularization(source_model, model)

        elif args.regularization_type == 'feature_map':
            backbone_regularization = BehavioralRegularization()

        # elif args.regularization_type == 'attention_feature_map':
        #     attention_file = os.path.join('delta_attention', f'{"GIN"}_{args.dataset}_{args.attention_file}')
        #     if os.path.exists(attention_file):
        #         print("Loading channel attention from", attention_file)
        #         attention = torch.load(attention_file)
        #         attention = [a.to(device) for a in attention]
        #     else:
        #         print('attention_file', attention_file)
        #         attention = calculate_channel_attention(train_dataset, return_layers, args)
        #         torch.save(attention, attention_file)

        #     backbone_regularization = AttentionBehavioralRegularization(attention)

        elif args.regularization_type == 'bss':
            bss_regularization = BatchSpectralShrinkage(k=args.k)
            # if args.debug:
            #     backbone_regularization = GTOTRegularization(order=args.gtot_order, args=args)
        # ------------------------------ end --------------------------------------------
        elif args.regularization_type == 'none':
            backbone_regularization = lambda x: x
            bss_regularization = lambda x: x
            pass
        else:
            raise NotImplementedError(args.regularization_type)

        head_regularization = L2Regularization(nn.ModuleList([model.graph_pred_linear]))

        train_acc_list = []
        val_acc_list = []
        test_acc_list = []

        args.filename = "Reg_FT" + "_" + args.dataset + "_" + args.split + "_" + "Fewshot_" + str(args.fewshot) + "_" + str(args.fewshot_num) + "_Reg_" + str(args.regularization_type) + "_m" + str(args.trade_off_backbone) + "_h" + str(args.trade_off_head)

        if not args.filename == "":
            fname = 'runs/Reg_FT_runseed' + str(args.runseed) + '/' + args.filename
            #delete the directory if there exists one
            # if os.path.exists(fname):
            #     shutil.rmtree(fname)
            #     print("removed the existing file.")
            writer = SummaryWriter(fname)
        
        best_func = min if args.regression else max
        best_val = 100 if args.regression else 0
        saved_model = model

        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train(args, epoch, model, device, train_loader, optimizer, weights_regularization, backbone_regularization, head_regularization, 
                                                target_getter, source_getter, bss_regularization)

            print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
            val_acc = eval(args, model, device, val_loader)
            test_acc = eval(args, model, device, test_loader)

            update = (best_val > val_acc) if args.regression else (best_val < val_acc)
            if update:
                print("update")
                best_val = val_acc
                saved_model = model

            print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)

            if not args.filename == "":
                writer.add_scalar('data/train auc', train_acc, epoch)
                writer.add_scalar('data/val auc', val_acc, epoch)
                writer.add_scalar('data/test auc', test_acc, epoch)

        print('Best epoch:', val_acc_list.index(best_func(val_acc_list)))
        print('Best auc: ', test_acc_list[val_acc_list.index(best_func(val_acc_list))])

        exp_path = os.getcwd() + '/{}_results/{}/'.format(args.input_model_file, args.dataset)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        df = pd.DataFrame({'train':train_acc_list,'valid':val_acc_list,'test':test_acc_list})
        df.to_csv(exp_path + args.filename + '_seed{}.csv'.format(args.runseed))

        score_list.append(test_acc_list[val_acc_list.index(best_func(val_acc_list))])
        logs = 'Dataset:{}, Split:{}, Fewshot_{}_{}, Seed:{}, Reg_{}_m{}_h{}, Best Epoch:{}, Best Acc:{:.5f}'.format(args.dataset, args.split, args.fewshot, args.fewshot_num, args.runseed, args.regularization_type, args.trade_off_backbone, args.trade_off_head, val_acc_list.index(best_func(val_acc_list)), test_acc_list[val_acc_list.index(best_func(val_acc_list))])
        with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
            f.write('\n')
            f.write(logs)
        torch.save(saved_model.state_dict(), exp_path + args.filename + '_seed{}.pth'.format(args.runseed))

        if not args.filename == "":
            writer.close()

    logs = 'Dataset:{}, Split:{}, Fewshot_{}_{}, All seed, Reg_{}_m{}_h{}, Best Acc:{:.5f}, std: {:.5f}'.format(args.dataset,args.split, args.fewshot, args.fewshot_num, args.regularization_type, args.trade_off_backbone, args.trade_off_head, np.mean(score_list), np.std(score_list))
    with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
        f.write('\n')
        f.write(logs)

if __name__ == "__main__":
    main()
