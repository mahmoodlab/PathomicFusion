import argparse
import os

import torch

### Parser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data/TCGA_GBMLGG', help="datasets")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_GBMLGG', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='exp_name', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='omic', help='mode')
    parser.add_argument('--model_name', type=str, default='omic', help='mode')
    parser.add_argument('--use_vgg_features', type=int, default=0, help='Use pretrained embeddings')
    parser.add_argument('--use_rnaseq', type=int, default=0, help='Use RNAseq data.')
    
    parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--useRNA', type=int, default=0) # Doesn't work at the moment...:(
    parser.add_argument('--useSN', type=int, default=1)
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--input_size_omic', type=int, default=80, help="input_size for omic vector")
    parser.add_argument('--input_size_path', type=int, default=512, help="input_size for path images")
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--save_at', type=int, default=20, help="adsfasdf")
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--print_every', default=0, type=int)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--finetune', default=1, type=int, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used for AdaBound')
    parser.add_argument('--reg_type', default='omic', type=str, help="regularization type")
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=25, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='start of epoch')
    parser.add_argument('--batch_size', type=int, default=32, help="Number of batches to train/test for. Default: 256")

    parser.add_argument('--lambda_cox', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=3e-4)
    parser.add_argument('--lambda_nll', type=float, default=1)


    parser.add_argument('--fusion_type', type=str, default="pofusion", help='concat | pofusion')
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--use_bilinear', type=int, default=1)
    parser.add_argument('--path_gate', type=int, default=1)
    parser.add_argument('--grph_gate', type=int, default=1)
    parser.add_argument('--omic_gate', type=int, default=1)
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--grph_dim', type=int, default=32)
    parser.add_argument('--omic_dim', type=int, default=32)
    parser.add_argument('--path_scale', type=int, default=1)
    parser.add_argument('--grph_scale', type=int, default=1)
    parser.add_argument('--omic_scale', type=int, default=1)
    parser.add_argument('--mmhid', type=int, default=64)

    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')
    parser.add_argument('--use_edges', default=1, type=float, help='Using edge_attr')
    parser.add_argument('--pooling_ratio', default=0.2, type=float, help='pooling ratio for SAGPOOl')
    parser.add_argument('--lr', default=2e-3, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--weight_decay', default=4e-4, type=float, help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--GNN', default='GCN', type=str, help='GCN | GAT | SAG. graph conv mode for pooling')
    parser.add_argument('--patience', default=0.005, type=float)
    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    opt = parse_gpuids(opt)
    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
