import argparse

parser = argparse.ArgumentParser()
# ========= Seed and basic info ==========
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=1)
parser.add_argument('--device', type=int, default=5)
parser.add_argument('--filename', type=str, default = 'debug', help='output filename')

# ========= Hyper-parameters ===========
parser.add_argument('--dataset', type=str, default='pcba')
parser.add_argument('--regression', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr_decay_step_size', type=int, default=150)
parser.add_argument('--lr_decay_factor', type=int, default=0.5)
parser.add_argument('--lr_cosine_length', type=int, default=400000, help='Cosine length if lr_schedule is cosine.')
parser.add_argument('--lr_warmup_steps', type=int, default=1e4, help='Warm-up Steps.')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patiance.')
parser.add_argument('--decay_patience', type=int, default=5, help='Scheduler decay patiance.')
parser.add_argument('--decay_factor', type=float, default=0.5, help='Scheduler decay patiance.')
parser.add_argument('--mask_rate', type=int, default=0.15)

# ======== Model configuration =========
parser.add_argument('--net2d', type=str, default='GIN')
parser.add_argument('--net_sm', type=str, default='transformer')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5) 
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--output_model_dir', type=str, default='./model_saved/')
parser.add_argument('--property', type=str, default='lumo', help='Regression Target')

# ========= Program viewing =========== 
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=False)

# ========= Neural Scaling Parameter =======
parser.add_argument('--selection', type=str, default='Uncertainty')
parser.add_argument('--selection_epochs', type=int, default=20)
parser.add_argument('--selection_lr', type=float, default=1e-3)
parser.add_argument('--selection_decay', type=float, default=0)
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--finetune_pruning', action='store_true')
parser.add_argument('--finetune_ratio', type=float, default=1)
parser.add_argument('--K', type=int, default=100)
parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")
parser.add_argument('--pretrain', type=bool, default=True)
parser.add_argument('--pretrain_model', type=str, default="graphmae_zinc")
parser.add_argument('--tune_option', type=str, default='linear_layer')
parser.add_argument('--fewshot', type=bool, default = False, help='whether few shot')
parser.add_argument('--fewshot_num', type=int, default = 50, help='few shot number for the labeled data')
parser.add_argument('--alpha', type=float, default = 1, help='alphas for WISE-FT')


# ========= Regularization based model parameter =======
parser.add_argument('--regularization_type', type=str, # choices=['l2_sp', 'feature_map', 'attention_feature_map',"none"],
                        default='l2_sp', help='fine tune regularization.')
parser.add_argument('--finetune_type', type=str, default='l2_sp', help='fine tune regularization.')  # choices=['delta', 'bitune', 'co_tune','l2_sp','none','bss'],
parser.add_argument('--norm_type', type=str, default='none', help='fine tune regularization.')
parser.add_argument('--trade_off_backbone', default=1, type=float, help='trade-off for backbone regularization')
parser.add_argument('--trade_off_head', default=1, type=float, help='trade-off for head regularization')
## bss
parser.add_argument('--trade_off_bss', default=1, type=float, help='trade-off for bss regularization')
parser.add_argument('-k', '--k', default=1, type=int, metavar='N', help='hyper-parameter for BSS loss')
parser.add_argument('--gtot_order', default=1, type=int, help='A^{k} in graph topology OT')
## for gtot
parser.add_argument('--train_radio', default=1.0, type=float, help='(train_set* train_radio) : val : test')
parser.add_argument('--dist_metric', default='norm_cosine', type=str, help='distance metric for optimal transport as cost matrix (cosine, norm_cosine)')
# parameters for calculating channel attention
parser.add_argument("--attention_file", type=str, default='channel_attention.pt', help="Where to save and load channel attention file.")
parser.add_argument("--data_path", type=str, default='./dataset', help="Where to save and load dataset.")
parser.add_argument('--attention-batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size for calculating channel attention (default: 32)')
parser.add_argument('--attention_epochs', default=50, type=int, metavar='N',
                    help='number of epochs to train for training before calculating channel weight')
parser.add_argument('--attention-lr-decay-epochs', default=30, type=int, metavar='N',
                    help='epochs to decay lr for training before calculating channel weight')
parser.add_argument('--attention_iteration_limit', default=50, type=int, metavar='N',
                    help='iteration limits for calculating channel attention, -1 means no limits')


parser.set_defaults(finetune_pruning=False)
# parser.set_defaults(pretrain=False)

args = parser.parse_args()
print('arguments\t', args)

