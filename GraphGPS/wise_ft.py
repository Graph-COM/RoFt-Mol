import datetime
import os
import torch
import logging
import pandas as pd
import time
import copy

import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
import argparse
import warnings
from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

from splitters import scaffold_split, random_split, random_scaffold_split, size_split
from torch_geometric.loader import DataLoader
from loader import MoleculeDataset
from graphgps.transform.transforms import pre_transform_in_memory
from graphgps.transform.posenc_stats import compute_posenc_stats
from functools import partial
from torch_geometric.graphgym.checkpoint import MODEL_STATE
from graphgps.train.custom_train import inference_only


# from graphgps.finetuning import load_pretrained_model_cfg, \
#     init_model_from_pretrained
# from graphgps.logger import create_logger


# torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
# torch.backends.cudnn.allow_tf32 = True  # Default True

# try:  # Define global config object
#     from yacs.config import CfgNode as CN
#     cfg = CN()
# except ImportError:
#     cfg = None
#     warnings.warn("Could not define global config object. Please install "
#                   "'yacs' via 'pip install yacs' in order to use GraphGym")

import os
import glob
import re

def verify_combined(combined: dict, ft_sd: dict, pt_sd: dict):
    """
    Prints:
      - whether combined == ft_sd (on all overlapping keys)
      - whether combined == pt_sd (on all overlapping keys)
      - for each 'post_mp' key, whether combined[k] == ft_sd[k] or pt_sd[k]
    """

    # 1) Global equality checks
    # Only compare keys that exist in all three dicts
    common_with_ft = [k for k in combined if k in ft_sd]
    common_with_pt = [k for k in combined if k in pt_sd]

    all_eq_ft = all(
        torch.equal(combined[k], ft_sd[k]) for k in common_with_ft
    )
    all_eq_pt = all(
        torch.equal(combined[k], pt_sd[k]) for k in common_with_pt
    )

    print(f"combined == finetuned on all overlapping keys? {all_eq_ft}")
    print(f"combined == pretrained on all overlapping keys? {all_eq_pt}")
    print()

    # 2) Per-key check for 'post_mp'
    post_keys = [k for k in combined if "post_mp" in k]
    if not post_keys:
        print("No keys containing 'post_mp' found in combined.")
        return

    print("Checking 'post_mp' keys:")
    for k in post_keys:
        eq_ft = torch.equal(combined[k], ft_sd.get(k, torch.tensor([])))
        eq_pt = torch.equal(combined[k], pt_sd.get(k, torch.tensor([])))
        print(f"  {k!r}: combined==ft? {eq_ft} ; combined==pt? {eq_pt}")

def detailed_verify_all(combined: dict, ft_sd: dict, pt_sd: dict):
    """
    For every key in the union of combined, ft_sd, and pt_sd, print:
      - if key exists in combined
      - if combined[key] == ft_sd[key] (when present)
      - if combined[key] == pt_sd[key] (when present)
    """
    all_keys = set(combined) | set(ft_sd) | set(pt_sd)
    for k in sorted(all_keys):
        in_comb = k in combined
        in_ft   = k in ft_sd
        in_pt   = k in pt_sd

        status = [f"in_combined={in_comb}", f"in_finetune={in_ft}", f"in_pretrain={in_pt}"]

        if in_comb and in_ft:
            status.append(f"==ft? {torch.equal(combined[k], ft_sd[k])}")
        if in_comb and in_pt:
            status.append(f"==pt? {torch.equal(combined[k], pt_sd[k])}")

        print(f"{k}: " + ", ".join(status))

def collect_best_ckpts(
    results_root: str,
    dataset_name: str,
    split: str,
    fewshot_bool: bool,
    fewshot_num: int,
    loss_fun: str
) -> list[str]:
    """
    Locate the FT_{dataset}_{split}_Fewshot_{fewshot_bool}_{fewshot_num}_fullFT_type_{loss_fun}_* folder
    under `results_root`, then in each of its subfolders pick the ckpt/*.ckpt file with the
    largest numeric basename.

    Returns:
        A list of full paths to the selected .ckpt files.
    """
    # 1) build glob pattern for the root folder
    pattern = (
        f"FT_{dataset_name}_{split}_Fewshot_{fewshot_bool}_{fewshot_num}"
        f"_fullFT_type_{loss_fun}_*"
    )
    candidate_roots = glob.glob(os.path.join(results_root, pattern))
    if not candidate_roots:
        raise FileNotFoundError(f"No folder matches pattern {pattern!r} under {results_root!r}")
    # (if there are multiple, pick the first; or adapt as you prefer)
    root_dir = candidate_roots[0]

    # 2) list immediate subdirectories
    subdirs = [
        d for d in glob.glob(os.path.join(root_dir, "*"))
        if os.path.isdir(d)
    ]

    best_ckpts = []
    for sub in subdirs:
        ckpt_dir = os.path.join(sub, "ckpt")
        if not os.path.isdir(ckpt_dir):
            continue

        # 3) find all .ckpt files
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if not ckpt_files:
            continue

        # 4) pick the one with the largest numeric prefix
        max_layer = -1
        best_file = None
        for fp in ckpt_files:
            name = os.path.basename(fp)
            # match files like "99.ckpt" or "3.ckpt" etc.
            m = re.match(r"^(\d+)\.ckpt$", name)
            if not m:
                continue
            layer = int(m.group(1))
            if layer > max_layer:
                max_layer = layer
                best_file = fp

        if best_file:
            best_ckpts.append(best_file)

    return best_ckpts

def compute_pe(cfg, dataset):
    pe_enabled_list = []
    # print(cfg)
    for key, pecfg in cfg.items():
        if key.startswith('posenc_'):
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                # if pecfg.kernel.times_func:
                pecfg.kernel.times = list(eval('range(1,17)'))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag, fewshot, fewshot_num, alpha, split):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    layer = 'WISE'
    run_name = "FT" + "_" + cfg.dataset.name + "_" + split + "_" + "Fewshot_" + str(fewshot) + "_" + str(fewshot_num) + "_"+ str(layer) + "_type_" + str(cfg.model.loss_fun) + "_alpha" + str(alpha)
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    # parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='GraphGym')
    # parser.add_argument('--fewshot_num', type=int, default=1,
    #                     help='The number of repeated jobs.')
    args = parse_args()
    print(args.opts)
    split = args.opts[0]
    fewshot = (args.opts[1]=='True')
    fewshot_num = int(args.opts[2])
    alpha = float(args.opts[3])
    # ft_type = 'l2_sp'
    # print(cfg)
    # Load config file
    set_cfg(cfg)
    args.opts = []
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, fewshot, fewshot_num, alpha, split)
    dump_cfg(cfg)
    cfg.optim.base_lr=0.001
    cfg.optim.lr_decay=0
    cfg.optim.max_epoch=100
    cfg.train.eval_period=5
    cfg.train.ckpt_period=5
    if cfg.dataset.name == "tox21":
        num_tasks = 12
    elif cfg.dataset.name == "hiv":
        num_tasks = 1
    elif cfg.dataset.name == "pcba":
        num_tasks = 128
    elif cfg.dataset.name == "muv":
        num_tasks = 17
    elif cfg.dataset.name == "bace":
        num_tasks = 1
    elif cfg.dataset.name == "bbbp":
        num_tasks = 1
    elif cfg.dataset.name == "toxcast":
        num_tasks = 617
    elif cfg.dataset.name == "sider":
        num_tasks = 27
    elif cfg.dataset.name == "clintox":
        num_tasks = 2
    elif cfg.dataset.name in ['esol', 'lipo', 'freesolv', 'malaria', 'cep', 'mpbp']:
        num_tasks = 1
    cfg.graphormer.out_dim = num_tasks
# Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dataset = MoleculeDataset("./dataset/" + cfg.dataset.name, dataset=cfg.dataset.name)
    print(dataset)
    
    # if split == "scaffold":
    smiles_list = pd.read_csv('./dataset/' + cfg.dataset.name + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, fewshot, fewshot_num, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = 42)
    print("scaffold")
    print('++++++++++', train_dataset)
    compute_pe(cfg, train_dataset)
    compute_pe(cfg, valid_dataset)
    compute_pe(cfg, test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 0)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers = 0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 0)
    loaders = [train_loader, val_loader, test_loader]
    
# Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                        f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        # loaders = create_loader()
        # loggers = create_logger()
        model = create_model()
        # logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)

        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )

        results_root  = "results"
        ckpt_list = collect_best_ckpts(
            results_root,
            cfg.dataset.name,
            split,
            fewshot,
            fewshot_num,
            cfg.model.loss_fun
        )
        print("Selected ckpt files:")
        ft_model_lists = []
        for ck in ckpt_list:
            print("  ", ck)
            ckpt = torch.load(ck)
            pretrained_dict = ckpt[MODEL_STATE]
            ft_model_lists.append(pretrained_dict)
        # print(model.state_dict())
        pt_sd = copy.deepcopy(model.state_dict())

        for ft_sd in ft_model_lists:
            combined = {}
            for k, pt_v in pt_sd.items():
                if k in ft_sd:
                    ft_v = ft_sd[k]
                    if "post_mp" in k:
                        combined[k] = ft_v.clone()
                    else:
                        combined[k] = (1 - alpha) * pt_v + alpha * ft_v
                else:
                    combined[k] = pt_v.clone()
            model.load_state_dict(combined)
            # verify_combined(combined, ft_sd, pt_sd)
            # detailed_verify_all(combined, ft_sd, pt_sd)

            for param in model.parameters():
                param.requires_grad = False
                model.eval()
            loggers = create_logger()
            # optimizer = create_optimizer(model.parameters(),
            #                             new_optimizer_config(cfg))
            # scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            inference_only(loggers, loaders, model)
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
        
        
