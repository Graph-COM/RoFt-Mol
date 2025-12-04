from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from rdkit.Chem import AllChem, RDKFingerprint
import math

from tqdm import tqdm
import random
import torch.nn.functional as F
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree 
from splitter import scaffold_split, random_split, imbalanced_split, size_split
from ftlib.reg_methods.bss import BatchSpectralShrinkage
from ftlib.reg_methods.delta import IntermediateLayerGetter, L2Regularization, get_attribute, BehavioralRegularization, AttentionBehavioralRegularization
from ftlib.reg_methods.delta import SPRegularization, FrobeniusRegularization
from ftlib.reg_methods.gtot_tuning import GTOTRegularization
from ftlib.reg_methods.meter import AverageMeter, ProgressMeter
from ftlib.reg_methods.eval import Meter
from tensorboardX import SummaryWriter
import os


from config import args
from datasets.molnet import MoleculeDataset
from model.gnn import GNN
from model.mlp import MLP
from utils import PrototypesGetHardExamples
from pruning.mini_kmeans import KMeans
import pruning.deepcore.methods as methods
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

def compute_degree(train_dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    return deg

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def save_model(save_best):
    dir_name = 'fingerprint/'
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))

            torch.save(model.state_dict(), args.output_model_dir \
                        + dir_name + 'transformer_{}_{}.pth'.format(args.dataset, args.finetune_ratio))

    return

def get_num_task(dataset):
    # Get output dimensions of different tasks
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'esol', 'lipo', 'freesolv', 'malaria', 'cep']:
        return 1
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'pcba':
        return 92
    raise ValueError('Invalid dataset name.')

def calculate_channel_attention(dataset, return_layers, args):
    device = args.device
    train_meter = Meter()

    model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(device)

    if args.pretrain:
            model_root = 'PubChem_Pretrained.pth'
            model.load_state_dict(torch.load(args.output_model_dir + model_root, map_location='cuda:0'))
            print('======= Model Loaded =======')
    classifier = output_layer

    data_loader = DataLoader(dataset, batch_size=args.attention_batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=False)

    model_param_group = []
    model_param_group.append({"params": model.parameters()})
    # if args.graph_pooling == "attention":
    #     model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": output_layer.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(
        math.log(0.1) / args.attention_lr_decay_epochs))

    # to save the change of loss when the output channel(every col in W) weights are masked.
    channel_weights = []
    for layer_id, name in enumerate(return_layers):
        layer = get_attribute(classifier, name)
        layer_channel_weight = [0] * layer.out_features
        channel_weights.append(layer_channel_weight)

    # train the classifier
    classifier.train()
    # classifier.gnn.requires_grad = False

    print("## Pretrain a classifier to calculate channel attention.")
    for epoch in range(args.attention_epochs):
        losses = AverageMeter('Loss', ':3.2f')
        cls_accs = AverageMeter('roc_auc_socre', ':3.1f')
        progress = ProgressMeter(
            len(data_loader),
            [losses, cls_accs],
            prefix="Epoch: [{}]".format(epoch))

        # for i, data in enumerate(data_loader):
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            h = global_mean_pool(model(batch), batch.batch)
            pred = output_layer(h)
            # loss = criterion(outputs, labels)

            y = batch.y.view(pred.shape).to(torch.float64)

            # Whether y is non-null or not.
            is_valid = y ** 2 > 0
            # Loss matrix
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            loss = torch.sum(loss_mat) / torch.sum(is_valid)

            train_meter.update((pred + 1) / 2, y, mask=is_valid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cls_acc = accuracy(outputs, labels)[0]
            cls_acc = np.mean(train_meter.compute_metric('roc_auc_score_finetune'))
            losses.update(loss.item(), batch.batch.size(0))
            cls_accs.update(cls_acc.item(), batch.batch.size(0))

            if i % args.print_freq == 0:
                progress.display(i)
        lr_scheduler.step()

    # calculate the channel attention
    print('Calculating channel attention.')
    classifier.eval()
    if args.attention_iteration_limit > 0:
        total_iteration = min(len(data_loader), args.attention_iteration_limit)
    else:
        total_iteration = len(args.data_loader)

    progress = ProgressMeter(
        total_iteration,
        [],
        prefix="Iteration: ")

    for i, batch in enumerate(data_loader):
        if i >= total_iteration:
            break
        batch = batch.to(device)
        h = global_mean_pool(model(batch), batch.batch)
        pred = output_layer(h)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat,
                               torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss_0 = torch.sum(loss_mat) / torch.sum(is_valid)

        if i % 20 == 0:
            progress.display(i)
        # for layer_id, name in enumerate(tqdm(return_layers)):
        for layer_id, name in enumerate(return_layers):
            layer = get_attribute(classifier, name)
            for j in range(layer.out_features):
                tmp = classifier.state_dict()[name + '.weight'][j,].clone()
                classifier.state_dict()[name + '.weight'][j,] = 0.0
                h = global_mean_pool(model(batch), batch.batch)
                pred = output_layer(h)
                loss_mat_1 = criterion(pred.double(), (y + 1.0) / 2)
                loss_mat_1 = torch.where(is_valid, loss_mat_1,
                                         torch.zeros(loss_mat_1.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss_1 = torch.sum(loss_mat_1) / torch.sum(is_valid)

                difference = loss_1 - loss_0
                difference = difference.detach().cpu().numpy().item()
                history_value = channel_weights[layer_id][j]
                # calculate mean vlaue of the increasement of loss
                channel_weights[layer_id][j] = 1.0 * (i * history_value + difference) / (i + 1)
                # recover the weight of model
                classifier.state_dict()[name + '.weight'][j,] = tmp

    channel_attention = []
    for weight in channel_weights:
        weight = np.array(weight)
        weight = (weight - np.mean(weight)) / np.std(weight)
        weight = torch.from_numpy(weight).float().to(device)
        channel_attention.append(F.softmax(weight / 5, dim=-1).detach())
    return channel_attention


# TODO: clean up
def train_general(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        h = global_mean_pool(model(batch), batch.batch)
        pred = output_layer(h)
        
        y = batch.y.view(pred.shape).to(torch.float64)
        # Whether y is non-null or not.
        
        # Loss matrix
        if args.regression:
            loss_mat = criterion_reg(pred.double(), y)
            loss_mat = torch.sqrt(loss_mat)
            is_valid = torch.ones_like(y).bool()
        else:
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            is_valid = y ** 2 > 0
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
        
        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    global optimal_loss 
    temp_loss = total_loss / len(loader)
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        # save_model(save_best=True)

    return total_loss / len(loader)

def train_epoch(args, model, device, loader, optimizer, weights_regularization, backbone_regularization, 
                head_regularization, target_getter,
                source_getter, bss_regularization, scheduler, epoch):
    model.train()

    meter = Meter()
    loss_epoch = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)
       
        intermediate_output_s, output_s = source_getter(batch)  # batch.batch is a column vector which maps each node to its respective graph in the batch
        intermediate_output_t, output_t = target_getter(batch)
        h = global_mean_pool(output_t, batch.batch)
        pred = output_layer(h)
        fea_s = global_mean_pool(output_s, batch.batch)
        fea = global_mean_pool(output_t, batch.batch)
            # intermediate_output_s
        
        h = global_mean_pool(model(batch), batch.batch)
        pred = output_layer(h)

        y = batch.y.view(pred.shape).to(torch.float64)
        # Whether y is non-null or not.
        
        # Loss matrix
        if args.regression:
            loss_mat = criterion_reg(pred.double(), y)
            loss_mat = torch.sqrt(loss_mat)
            is_valid = torch.ones_like(y).bool()
        else:
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            is_valid = y ** 2 > 0
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
        cls_loss = torch.sum(loss_mat) / torch.sum(is_valid)

        meter.update(pred, y, mask=is_valid)

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
            fea = fea if fea is not None else global_mean_pool(model(batch), batch.batch)
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

        loss_epoch.append(cls_loss.item())
    
    avg_loss = sum(loss_epoch) / len(loss_epoch)

    if scheduler is not None: scheduler.step()
    print(f'{"vanilla model || " if fea is None and args.norm_type == "none" else ""} '
        f'cls_loss:{avg_loss:.5f}, loss_reg_backbone: {args.trade_off_backbone * loss_reg_backbone:.5f} loss_reg_head:'
        f' {args.trade_off_head * loss_reg_head:.5f} bss_los: {args.trade_off_bss * loss_bss:.5f} ' + print_str)
    try:
        print('num_oversmooth:', backbone_regularization.num_oversmooth, end=' || ')
        backbone_regularization.num_oversmooth = 0
    except:
        pass

    # global optimal_loss 
    # temp_loss = sum(loss_epoch) / len(loader)
    # if temp_loss < optimal_loss:
    #     optimal_loss = temp_loss

       

    return avg_loss

def eval_general(model, device, loader):
    model.eval()
    y_true, y_scores = [], []
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            h = global_mean_pool(model(batch), batch.batch)
            pred = output_layer(h)
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)


    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if args.dataset == 'pcba':
        ap_list = []

        for i in range(y_true.shape[1]):
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                # ignore nan values
                is_valid = is_valid = y_true[:, i] ** 2 > 0
                ap = average_precision_score(y_true[is_valid, i], y_scores[is_valid, i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')
        return sum(ap_list) / len(ap_list), total_loss / len(loader), ap_list

    else:
        roc_list = []
        if args.regression:
            roc_list.append(math.sqrt(mean_squared_error(y_true, y_scores)))
        else:
            for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                    is_valid = y_true[:, i] ** 2 > 0
                    roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
                else:
                    print('{} is invalid'.format(i))

            if len(roc_list) < y_true.shape[1]:
                print(len(roc_list))
                print('Some target is missing!')
                print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

        return sum(roc_list) / len(roc_list), total_loss / len(loader), roc_list


if __name__ == '__main__':
    torch.set_num_threads(10)
    score_list = []
    allseed = [0, 1, 2, 3, 4]
    for seed in allseed:
        args.runseed = seed
        seed_all(args.runseed)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)

        # Bunch of classification tasks
        num_tasks = get_num_task(args.dataset)
        dataset_folder = './datasets/molecule_net/'
        dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
        args.regression = True if args.dataset in ['esol', 'lipo', 'freesolv', 'malaria', 'cep'] else False

        print(dataset)
        print('=============== Statistics ==============')
        print('Avg degree:{}'.format(torch.sum(degree(dataset.data.edge_index[0])).item()/dataset.data.x.shape[0]))
        print('Avg atoms:{}'.format(dataset.data.x.shape[0]/(dataset.data.y.shape[0]/num_tasks)))
        print('Avg bond:{}'.format((dataset.data.edge_index.shape[1]/2)/(dataset.data.y.shape[0]/num_tasks)))

        eval_metric = roc_auc_score

        if args.split == 'scaffold':
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                    header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = scaffold_split(
                dataset, smiles_list, args.fewshot, args.fewshot_num, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=args.seed, return_smiles=True)
            print('split via scaffold')
        elif args.split == "size":
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                    header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = size_split(
                dataset, smiles_list, args.fewshot, args.fewshot_num, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=args.seed, return_smiles=True)
        elif args.split == 'random':
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                    header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles),_ = random_split(
                dataset, args.fewshot, args.fewshot_num, null_value=0, frac_train=0.8, frac_valid=0.1,
                frac_test=0.1, seed=args.seed, smiles_list=smiles_list)
            print('randomly split')
        elif args.split == 'imbalanced':
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                    header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = imbalanced_split(
                dataset, null_value=0, frac_train=0.7, frac_valid=0.15,
                frac_test=0.15, seed=args.seed, smiles_list=smiles_list)
            print('imbalanced split')
            
        else:
            raise ValueError('Invalid split option.')
        print(train_dataset[0])
        print('Training data length: {}'.format(len(train_smiles)))
        finetune_num = int(args.finetune_ratio * len(train_smiles))

        if args.finetune_pruning is True and args.selection == 'Kmeans':
            print('# of SMILES: {}'.format(len(train_smiles)))
            print('===== Converting smiles to mols =====')
            mols = [AllChem.MolFromSmiles(s) for s in tqdm(train_smiles)]
            print('===== Processing fingerprint =====')
            fps = [torch.tensor(RDKFingerprint(mol), dtype=torch.float).unsqueeze(0) for mol in tqdm(mols)]
            fps = torch.cat(fps, dim=0)
            print('====== Fingerprint Finish ! ! ! =======')

            if len(train_smiles) > 1e5: K = 1000
            else: K = 100
            kmeans = KMeans(n_clusters=K, device=device)
            centers = kmeans.fit_predict(fps)
            scores, cluster_labels = kmeans.predict(fps)
            scores = scores.cpu().detach()
            cluster_labels = cluster_labels.cpu().detach()
            ids = PrototypesGetHardExamples(scores, cluster_labels, range(len(fps)), return_size=finetune_num)
            train_dataset = train_dataset[ids]

        elif args.finetune_pruning is True:
            selection_args = dict(epochs=args.selection_epochs,
                                    selection_method=args.uncertainty,
                                    num_tasks=num_tasks)
            method = methods.__dict__[args.selection](dst_train=train_dataset, args=args, fraction=args.finetune_ratio, 
                    random_seed=args.runseed, device = device, **selection_args)
            subset = method.select()
            print(len(subset["indices"]))
            train_dataset = train_dataset[subset["indices"]]
        
        else:
            num_mols = len(train_dataset)
            random.seed(args.runseed)
            all_idx = list(range(num_mols))
            random.shuffle(all_idx)
            ids = all_idx[:int(args.finetune_ratio * num_mols)]
            train_dataset = train_dataset[ids]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)            

        # set up model 
        model_param_group = []
        model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(device)
        source_model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(device)

        output_layer = MLP(in_channels=args.emb_dim, hidden_channels=args.emb_dim, 
                            out_channels=num_tasks, num_layers=1, dropout=0).to(device)
        source_output_layer = MLP(in_channels=args.emb_dim, hidden_channels=args.emb_dim, 
                            out_channels=num_tasks, num_layers=1, dropout=0).to(device)
        
        if args.pretrain_model == "None":
            args.pretrain = False 
        
        print(args.pretrain_model)
        print(args.pretrain)

        if args.pretrain:
            if args.pretrain_model == "graphmae_zinc":
                model_root = 'pretrained.pth'
            else:
                model_root = 'PubChem_Pretrained.pth'
            
            model.load_state_dict(torch.load(args.output_model_dir + model_root, map_location='cuda:0'))
            print('======= Model Loaded =======')
        # model_param_group.append({'params': output_layer.parameters(),'lr': args.lr})
        # model_param_group.append({'params': model.parameters(), 'lr': args.lr})
        # total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in output_layer.parameters())
        # print(f"Number of parameters: {total_params}")
        for param in source_model.parameters():
            param.requires_grad = False
            source_model.eval()

        for param in source_output_layer.parameters():
            param.requires_grad = False
            source_output_layer.eval()

        print(model)

        # create intermediate layer getter
        return_layers = ['gnns.4.mlp.2'] # for GIN
        source_getter = IntermediateLayerGetter(source_model, return_layers=return_layers)
        target_getter = IntermediateLayerGetter(model, return_layers=return_layers)

        optimizer = optim.Adam(list(model.parameters()) + list(output_layer.parameters()), lr=args.lr, weight_decay=args.decay)

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

        elif args.regularization_type == 'attention_feature_map':
            attention_file = os.path.join('delta_attention', f'{"GIN"}_{args.dataset}_{args.attention_file}')
            if os.path.exists(attention_file):
                print("Loading channel attention from", attention_file)
                attention = torch.load(attention_file)
                attention = [a.to(device) for a in attention]
            else:
                print('attention_file', attention_file)
                attention = calculate_channel_attention(train_dataset, return_layers, args)
                torch.save(attention, attention_file)

            backbone_regularization = AttentionBehavioralRegularization(attention)

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

        head_regularization = L2Regularization(nn.ModuleList([output_layer]))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=6,
                                                            verbose=False,
                                                            threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                            min_lr=1e-8,
                                                            eps=1e-08)

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        criterion_reg = nn.MSELoss()
        best_func = min if args.regression else max
        train_roc_list, val_roc_list, test_roc_list = [], [], []
        train_loss_list, val_loss_list, test_loss_list = [], [], []
        roc_lists = []
        best_val_roc, best_val_idx = -1, 0
        optimal_loss = 1e10
        es = 0
        best_val = 100 if args.regression else 0
        saved_model = model

        train_func = train_general
        eval_func = eval_general

        args.filename = "Reg_FT" + "_" + args.dataset + "_" + args.split + "_" + "Fewshot_" + str(args.fewshot) + "_" + str(args.fewshot_num) + "_Reg_" + str(args.regularization_type) + "_m" + str(args.trade_off_backbone) + "_h" + str(args.trade_off_head)

        if not args.filename == "":
            fname = 'runs/Reg_FT_runseed' + str(args.runseed) + '/' + args.filename
            #delete the directory if there exists one
            # if os.path.exists(fname):
            #     shutil.rmtree(fname)
            #     print("removed the existing file.")
            writer = SummaryWriter(fname)

        for epoch in range(1, args.epochs + 1):
            loss_acc = train_epoch(args, model, device, train_loader, optimizer, weights_regularization, backbone_regularization, head_regularization, 
                                                target_getter, source_getter, bss_regularization, scheduler=None, epoch=epoch)
            # loss_acc = train_func(model, device, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            # train_roc = train_loss = 0
            train_roc, train_loss, _ = eval_func(model, device, train_loader)
            val_roc, val_loss, _ = eval_func(model, device, val_loader)
            test_roc, test_loss, roc_list = eval_func(model, device, test_loader)

            train_roc_list.append(train_roc)
            val_roc_list.append(val_roc)
            test_roc_list.append(test_roc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)
            roc_lists.append(roc_list)
            print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
            print()

            update = (best_val > val_roc) if args.regression else (best_val < val_roc)
            if update:
                print("update")
                best_val = val_roc
                saved_model = model

            if not args.filename == "":
                writer.add_scalar('data/train auc', train_roc, epoch)
                writer.add_scalar('data/val auc', val_roc, epoch)
                writer.add_scalar('data/test auc', test_roc, epoch)

        # print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))
        # print('loss train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_loss_list[best_val_idx], val_loss_list[best_val_idx], test_loss_list[best_val_idx]))
        # print('single tasks roc list:{}'.format(roc_lists[best_val_idx]))

        exp_path = os.getcwd() + '/finetune_results/{}/'.format(args.dataset)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        df = pd.DataFrame({'train':train_roc_list,'valid':val_roc_list,'test':test_roc_list})
        df.to_csv(exp_path + args.filename + '_seed{}.csv'.format(args.runseed))

        score_list.append(test_roc_list[val_roc_list.index(best_func(val_roc_list))])
        logs = 'Dataset:{}, Split:{}, Fewshot_{}_{}, Seed:{}, Reg_{}_m{}_h{}, Best Epoch:{}, Best Acc:{:.5f}'.format(args.dataset, args.split, args.fewshot, args.fewshot_num, args.runseed, 
                                                                                                                     args.regularization_type, args.trade_off_backbone, args.trade_off_head,
                                                                                                                     val_roc_list.index(best_func(val_roc_list)), test_roc_list[val_roc_list.index(best_func(val_roc_list))])
        with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
            f.write('\n')
            f.write(logs)
        torch.save(saved_model.state_dict(), exp_path + args.filename + '_seed{}.pth'.format(args.runseed))

        if not args.filename == "":
            writer.close()
    
    logs = 'Dataset:{}, Split:{}, Fewshot_{}_{}, All seed, Reg_{}_m{}_h{}, Best Acc:{:.5f}, std: {:.5f}'.format(args.dataset, args.split, args.fewshot, args.fewshot_num, 
                                                                                                                args.regularization_type, args.trade_off_backbone, args.trade_off_head,
                                                                                                                np.mean(score_list), np.std(score_list))
    with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
        f.write('\n')
        f.write(logs)


    
