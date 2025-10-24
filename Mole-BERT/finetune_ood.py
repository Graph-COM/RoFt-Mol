import torch
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import math

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import itertools

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from splitters import scaffold_split, random_split, random_scaffold_split, size_split
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion_reg = nn.MSELoss()

def train(args, epoch, model, device, loader, optimizer):
    model.train()
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
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
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")


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
    error = mean_absolute_error if args.dataset == "mpbp" else mean_squared_error
    # error = mean_squared_error
    print(error)
    if args.regression:
        roc_list.append(math.sqrt(error(y_true, y_scores)))
        # roc_list.append(error(y_true, y_scores))
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
    parser.add_argument('--dataset', type=str, default = 'clintox', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'Mole-BERT', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = 'debug', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--tune_option', type=str, default = "all", help='number of workers for dataset loading')
    parser.add_argument('--regression', type=bool, default = False, help='whether regression task')
    parser.add_argument('--fewshot', type=bool, default = False, help='whether few shot')
    parser.add_argument('--fewshot_num', type=int, default = 50, help='few shot number for the labeled data')

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
            print(criterion_reg)

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

        print(train_dataset[0])

        #set up model
        model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        if not args.input_model_file == "None":
            print('Not from scratch')
            model.from_pretrained('model_gin/{}.pth'.format(args.input_model_file))
        
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

        tune_params_dict = {
            "all": [model.gnn.parameters(), model.graph_pred_linear.parameters()],
            "first_layer": [
                model.gnn.gnns[0].parameters()
            ],
            "second_layer": [
                model.gnn.gnns[1].parameters()
            ],
            "third_layer": [
                model.gnn.gnns[2].parameters()
            ],
            "fourth_layer": [
                model.gnn.gnns[3].parameters()
            ],
            "fifth_layer": [
                model.gnn.gnns[4].parameters()
            ],
            "first_linear": [
                model.gnn.gnns[0].parameters(),model.graph_pred_linear.parameters()
            ],
            "second_linear": [
                model.gnn.gnns[1].parameters(),model.graph_pred_linear.parameters()
            ],
            "third_linear": [
                model.gnn.gnns[2].parameters(),model.graph_pred_linear.parameters()
            ],
            "fourth_linear": [
                model.gnn.gnns[3].parameters(),model.graph_pred_linear.parameters()
            ],
            "fifth_linear": [
                model.gnn.gnns[4].parameters(),model.graph_pred_linear.parameters()
            ],
            "linear_layer": [model.graph_pred_linear.parameters()],
        }
        #set up optimizer
        #different learning rate for different part of GNN
        # model_param_group = []
        # model_param_group.append({"params": model.gnn.parameters()})
        # if args.graph_pooling == "attention":
        #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})

        params_list = list(itertools.chain(*tune_params_dict[args.tune_option]))
        optimizer = optim.Adam(params_list, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        train_acc_list = []
        val_acc_list = []
        test_acc_list = []

        args.filename = "Fewshot" + "_" + args.dataset + "_" + args.tune_option + "_" + args.split + "_" + "Fewshot_" + str(args.fewshot) + "_" + str(args.fewshot_num)

        if not args.filename == "":
            fname = 'runs/finetune_fewshot_cls_runseed' + str(args.runseed) + '/' + args.filename
            #delete the directory if there exists one
            # if os.path.exists(fname):
            #     shutil.rmtree(fname)
            #     print("removed the existing file.")
            writer = SummaryWriter(fname)
        
        exp_path = os.getcwd() + '/{}_results/{}/'.format(args.input_model_file, args.dataset)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        best_func = min if args.regression else max
        best_val = 10000 if args.regression else 0
        saved_model = {}

        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train(args, epoch, model, device, train_loader, optimizer)

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
                # saved_model = model.state_dict()
                torch.save(model.state_dict(), exp_path + args.filename + '_seed{}.pth'.format(args.runseed))
                # if all(saved_model[key] == model.state_dict()[key] for key in model.state_dict().keys()):
                #     print("Dictionaries are equal")
                # else:
                #     print("Dictionaries are not equal")

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


        df = pd.DataFrame({'train':train_acc_list,'valid':val_acc_list,'test':test_acc_list})
        df.to_csv(exp_path + args.filename + '_seed{}.csv'.format(args.runseed))

        score_list.append(test_acc_list[val_acc_list.index(best_func(val_acc_list))])
        logs = 'Dataset:{}, Split:{}, Fewshot_{}_{}, Seed:{}, Tune layers:{}, Best Epoch:{}, Best Acc:{:.5f}'.format(args.dataset, args.split, args.fewshot, args.fewshot_num, args.runseed, args.tune_option, val_acc_list.index(best_func(val_acc_list)), test_acc_list[val_acc_list.index(best_func(val_acc_list))])
        with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
            f.write('\n')
            f.write(logs)
        # torch.save(saved_model, exp_path + args.filename + '_seed{}.pth'.format(args.runseed))

        if not args.filename == "":
            writer.close()

    logs = 'Dataset:{}, Split:{}, Fewshot_{}_{}, All seed, Tune layers:{}, Best Acc:{:.5f}, std: {:.5f}'.format(args.dataset,args.split, args.fewshot, args.fewshot_num, args.tune_option, np.mean(score_list), np.std(score_list))
    with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
        f.write('\n')
        f.write(logs)

if __name__ == "__main__":
    main()
