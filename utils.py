# Base / Native
import math
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Numerical / Array
import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from imblearn.over_sampling import RandomOverSampler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from PIL import Image
import pylab
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score, auc, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from scipy import interp
mpl.rcParams['axes.linewidth'] = 3 #set the value globally

# Torch
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch



################
# Regularization
################
def regularize_weights(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


def regularize_path_weights(model, reg_type=None):
    l1_reg = None
    
    for W in model.module.classifier.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    for W in model.module.linear.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    return l1_reg


def regularize_MM_weights(model, reg_type=None):
    l1_reg = None

    if model.module.__hasattr__('omic_net'):
        for W in model.module.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_h_path'):
        for W in model.module.linear_h_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_h_omic'):
        for W in model.module.linear_h_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_h_grph'):
        for W in model.module.linear_h_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_z_path'):
        for W in model.module.linear_z_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_z_omic'):
        for W in model.module.linear_z_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_z_grph'):
        for W in model.module.linear_z_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_o_path'):
        for W in model.module.linear_o_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_o_omic'):
        for W in model.module.linear_o_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_o_grph'):
        for W in model.module.linear_o_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('encoder1'):
        for W in model.module.encoder1.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('encoder2'):
        for W in model.module.encoder2.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('classifier'):
        for W in model.module.classifier.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
        
    return l1_reg


def regularize_MM_omic(model, reg_type=None):
    l1_reg = None

    if model.module.__hasattr__('omic_net'):
        for W in model.module.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    return l1_reg



################
# Network Initialization
################
def init_weights(net, init_type='orthogonal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)           # multi-GPUs

    if init_type != 'max' and init_type != 'none':
        print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")
    return net



################
# Freeze / Unfreeze
################
def unfreeze_unimodal(opt, model, epoch):
    if opt.mode == 'graphomic':
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == 'pathomic':
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == 'pathgraph':
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "pathgraphomic":
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "omicomic":
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == "graphgraph":
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def print_if_frozen(module):
    for idx, child in enumerate(module.children()):
        for param in child.parameters():
            if param.requires_grad == True:
                print("Learnable!!! %d:" % idx, child)
            else:
                print("Still Frozen %d:" % idx, child)


def unfreeze_vgg_features(model, epoch):
    epoch_schedule = {30:45}
    unfreeze_index = epoch_schedule[epoch]
    for idx, child in enumerate(model.features.children()):
        if idx > unfreeze_index:
            print("Unfreezing %d:" %idx, child)
            for param in child.parameters(): 
                param.requires_grad = True
        else:
            print("Still Frozen %d:" %idx, child)
            continue



################
# Collate Utils
################
def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)    
    transposed = zip(*batch)
    return [Batch.from_data_list(samples, []) if type(samples[0]) is torch_geometric.data.data.Data else default_collate(samples) for samples in transposed]



################
# Survival Utils
################
def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)


def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))



################
# Data Utils
################
def addHistomolecularSubtype(data):
    """
    Molecular Subtype: IDHwt == 0, IDHmut-non-codel == 1, IDHmut-codel == 2
    Histology Subtype: astrocytoma == 0, oligoastrocytoma == 1, oligodendroglioma == 2, glioblastoma == 3
    """
    subtyped_data = data.copy()
    subtyped_data.insert(loc=0, column='Histomolecular subtype', value=np.ones(len(data)))
    idhwt_ATC = np.logical_and(data['Molecular subtype'] == 0, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhwt_ATC, 'Histomolecular subtype'] = 'idhwt_ATC'
    
    idhmut_ATC = np.logical_and(data['Molecular subtype'] == 1, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhmut_ATC, 'Histomolecular subtype'] = 'idhmut_ATC'
    
    ODG = np.logical_and(data['Molecular subtype'] == 2, data['Histology'] == 2)
    subtyped_data.loc[ODG, 'Histomolecular subtype'] = 'ODG'
    return subtyped_data


def changeHistomolecularSubtype(data):
    """
    Molecular Subtype: IDHwt == 0, IDHmut-non-codel == 1, IDHmut-codel == 2
    Histology Subtype: astrocytoma == 0, oligoastrocytoma == 1, oligodendroglioma == 2, glioblastoma == 3
    """
    data = data.drop(['Histomolecular subtype'], axis=1)
    subtyped_data = data.copy()
    subtyped_data.insert(loc=0, column='Histomolecular subtype', value=np.ones(len(data)))
    idhwt_ATC = np.logical_and(data['Molecular subtype'] == 0, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhwt_ATC, 'Histomolecular subtype'] = 'idhwt_ATC'
    
    idhmut_ATC = np.logical_and(data['Molecular subtype'] == 1, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhmut_ATC, 'Histomolecular subtype'] = 'idhmut_ATC'
    
    ODG = np.logical_and(data['Molecular subtype'] == 2, data['Histology'] == 2)
    subtyped_data.loc[ODG, 'Histomolecular subtype'] = 'ODG'
    return subtyped_data


def getCleanAllDataset(dataroot='./data/TCGA_GBMLGG/', ignore_missing_moltype=False, ignore_missing_histype=False, use_rnaseq=False):
    ### 1. Joining all_datasets.csv with grade data. Looks at columns with misisng samples
    metadata = ['Histology', 'Grade', 'Molecular subtype', 'TCGA ID', 'censored', 'Survival months']
    all_dataset = pd.read_csv(os.path.join(dataroot, 'all_dataset.csv')).drop('indexes', axis=1)
    all_dataset.index = all_dataset['TCGA ID']

    all_grade = pd.read_csv(os.path.join(dataroot, 'grade_data.csv'))
    all_grade['Histology'] = all_grade['Histology'].str.replace('astrocytoma (glioblastoma)', 'glioblastoma', regex=False)
    all_grade.index = all_grade['TCGA ID']
    assert pd.Series(all_dataset.index).equals(pd.Series(sorted(all_grade.index)))

    all_dataset = all_dataset.join(all_grade[['Histology', 'Grade', 'Molecular subtype']], how='inner')
    cols = all_dataset.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    all_dataset = all_dataset[cols]

    if use_rnaseq:
        gbm = pd.read_csv(os.path.join(dataroot, 'mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt'), sep='\t', skiprows=1, index_col=0)
        lgg = pd.read_csv(os.path.join(dataroot, 'mRNA_Expression_Zscores_RSEM.txt'), sep='\t', skiprows=1, index_col=0)
        gbm = gbm[gbm.columns[~gbm.isnull().all()]]
        lgg = lgg[lgg.columns[~lgg.isnull().all()]]
        glioma_RNAseq = gbm.join(lgg, how='inner').T
        glioma_RNAseq = glioma_RNAseq.dropna(axis=1)
        glioma_RNAseq.columns = [gene+'_rnaseq' for gene in glioma_RNAseq.columns]
        glioma_RNAseq.index = [patname[:12] for patname in glioma_RNAseq.index]
        glioma_RNAseq = glioma_RNAseq.iloc[~glioma_RNAseq.index.duplicated()]
        glioma_RNAseq.index.name = 'TCGA ID'
        all_dataset = all_dataset.join(glioma_RNAseq, how='inner')

    pat_missing_moltype = all_dataset[all_dataset['Molecular subtype'].isna()].index
    pat_missing_idh = all_dataset[all_dataset['idh mutation'].isna()].index
    pat_missing_1p19q = all_dataset[all_dataset['codeletion'].isna()].index
    print("# Missing Molecular Subtype:", len(pat_missing_moltype))
    print("# Missing IDH Mutation:", len(pat_missing_idh))
    print("# Missing 1p19q Codeletion:", len(pat_missing_1p19q))
    assert pat_missing_moltype.equals(pat_missing_idh)
    assert pat_missing_moltype.equals(pat_missing_1p19q)
    pat_missing_grade =  all_dataset[all_dataset['Grade'].isna()].index
    pat_missing_histype = all_dataset[all_dataset['Histology'].isna()].index
    print("# Missing Histological Subtype:", len(pat_missing_histype))
    print("# Missing Grade:", len(pat_missing_grade))
    assert pat_missing_histype.equals(pat_missing_grade)

    ### 2. Impute Missing Genomic Data: Removes patients with missing molecular subtype / idh mutation / 1p19q. Else imputes with median value of each column. Fills missing Molecular subtype with "Missing"
    if ignore_missing_moltype: 
        all_dataset = all_dataset[all_dataset['Molecular subtype'].isna() == False]
    for col in all_dataset.drop(metadata, axis=1).columns:
        all_dataset['Molecular subtype'] = all_dataset['Molecular subtype'].fillna('Missing')
        all_dataset[col] = all_dataset[col].fillna(all_dataset[col].median())

    ### 3. Impute Missing Histological Data: Removes patients with missing histological subtype / grade. Else imputes with "missing" / grade -1
    if ignore_missing_histype: 
        all_dataset = all_dataset[all_dataset['Histology'].isna() == False]
    else:
        all_dataset['Grade'] = all_dataset['Grade'].fillna(1)
        all_dataset['Histology'] = all_dataset['Histology'].fillna('Missing')
    all_dataset['Grade'] = all_dataset['Grade'] - 2

    ### 4. Adds Histomolecular subtype
    ms2int = {'Missing':-1, 'IDHwt':0, 'IDHmut-non-codel':1, 'IDHmut-codel':2}
    all_dataset[['Molecular subtype']] = all_dataset[['Molecular subtype']].applymap(lambda s: ms2int.get(s) if s in ms2int else s)
    hs2int = {'Missing':-1, 'astrocytoma':0, 'oligoastrocytoma':1, 'oligodendroglioma':2, 'glioblastoma':3}
    all_dataset[['Histology']] = all_dataset[['Histology']].applymap(lambda s: hs2int.get(s) if s in hs2int else s)
    all_dataset = addHistomolecularSubtype(all_dataset)
    metadata.extend(['Histomolecular subtype'])
    all_dataset['censored'] = 1 - all_dataset['censored']
    return metadata, all_dataset



################
# Analysis Utils
################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hazard2grade(hazard, p):
    if hazard < p[0]:
        return 0
    elif hazard < p[1]:
        return 1
    return 2


def p(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'p%s' % n
    return percentile_


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def CI_pm(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return str("{0:.4f} ± ".format(m) + "{0:.3f}".format(h))


def CI_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return str("{0:.3f}, ".format(m-h) + "{0:.3f}".format(m+h))


def poolSurvTestPD(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', split='test', zscore=False, agg_type='Hazard_mean'):
    all_dataset_regstrd_pooled = []    
    ignore_missing_moltype = 1 if 'omic' in model else 0
    ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if ((('path' in model) or ('graph' in model)) and ('cox' not in model)) else ('_', 'all_st', 0)
    use_rnaseq = '_rnaseq' if ('rnaseq' in ckpt_name and 'path' != model and 'pathpath' not in model and 'graph' != model and 'graphgraph' not in model) else ''

    for k in range(1,16):
        pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
        
        if 'cox' not in model:
            surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
            surv_all.columns = ['Hazard', 'Survival months', 'censored', 'Grade']
            data_cv = pickle.load(open('./data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq), 'rb'))
            data_cv_splits = data_cv['cv_splits']
            data_cv_split_k = data_cv_splits[k]
            assert np.all(data_cv_split_k[split]['t'] == pred[1]) # Data is correctly registered
            all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
            all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
            assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
            assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
            assert np.all(np.array(all_dataset_regstrd['Grade']) == pred[4])
            all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))
            all_dataset_regstrd.index.name = 'TCGA ID'
            hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max, p(0.25), p(0.75)]})
            hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
            hazard_agg = hazard_agg[[agg_type]]
            hazard_agg.columns = ['Hazard']
            pred = hazard_agg.join(all_dataset, how='inner')

        if zscore: pred['Hazard'] = scipy.stats.zscore(np.array(pred['Hazard']))
        all_dataset_regstrd_pooled.append(pred)

    all_dataset_regstrd_pooled = pd.concat(all_dataset_regstrd_pooled)
    all_dataset_regstrd_pooled = changeHistomolecularSubtype(all_dataset_regstrd_pooled)
    return all_dataset_regstrd_pooled


def getAggHazardCV(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', split='test', agg_type='Hazard_mean'):
    result = []
    
    ignore_missing_moltype = 1 if 'omic' in model else 0
    ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)
    use_rnaseq = '_rnaseq' if ('rnaseq' in ckpt_name and 'path' != model and 'pathpath' not in model and 'graph' != model and 'graphgraph' not in model) else ''

    for k in range(1,16):
        pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
        surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
        surv_all.columns = ['Hazard', 'Survival months', 'censored', 'Grade']
        data_cv = pickle.load(open('./data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq), 'rb'))
        data_cv_splits = data_cv['cv_splits']
        data_cv_split_k = data_cv_splits[k]
        assert np.all(data_cv_split_k[split]['t'] == pred[1]) # Data is correctly registered
        all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
        all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
        assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
        assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
        assert np.all(np.array(all_dataset_regstrd['Grade']) == pred[4])
        all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))
        all_dataset_regstrd.index.name = 'TCGA ID'
        hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', max, p(0.75)]})
        hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
        hazard_agg = hazard_agg[[agg_type]]
        hazard_agg.columns = ['Hazard']
        all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')
        cin = CIndex_lifeline(all_dataset_hazard['Hazard'], all_dataset_hazard['censored'], all_dataset_hazard['Survival months'])
        result.append(cin)
        
    return result


def calcGradMetrics(ckpt_name='./checkpoints/grad_15/', model='pathgraphomic_fusion', split='test', avg='micro'):
    auc_all = []
    ap_all = []
    f1_all = []
    f1_gradeIV_all = []
    
    ignore_missing_moltype = 1 if 'omic' in model else 0
    ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)
    
    for k in range(1,16):
        pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
        grade_pred, grade = np.array(pred[3]), np.array(pred[4])
        enc = LabelBinarizer()
        enc.fit(grade)
        grade_oh = enc.transform(grade)
        rocauc = roc_auc_score(grade_oh, grade_pred, avg)
        ap = average_precision_score(grade_oh, grade_pred, average=avg)
        f1 = f1_score(grade_pred.argmax(axis=1), grade, average=avg)
        f1_gradeIV = f1_score(grade_pred.argmax(axis=1), grade, average=None)[2]
        
        auc_all.append(rocauc)
        ap_all.append(ap)
        f1_all.append(f1)
        f1_gradeIV_all.append(f1_gradeIV)
        
    return np.array([CI_pm(auc_all), CI_pm(ap_all), CI_pm(f1_all), CI_pm(f1_gradeIV_all)])



################
# Plot Utils
################
def makeKaplanMeierPlot(ckpt_name='./checkpoints/surv_15_rnaseq/', model='omic', split='test', zscore=False, agg_type='Hazard_mean'):
    def hazard2KMCurve(data, subtype):
        p = np.percentile(data['Hazard'], [33, 66])
        if p[0] == p[1]: p[0] = 2.99997
        data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
        kmf_pred = lifelines.KaplanMeierFitter()
        kmf_gt = lifelines.KaplanMeierFitter()

        def get_name(model):
            mode2name = {'pathgraphomic':'Pathomic F.', 'pathomic':'Pathomic F.', 'graphomic':'Pathomic F.', 'path':'Histology CNN', 'graph':'Histology GCN', 'omic':'Genomic SNN'}
            for mode in mode2name.keys():
                if mode in model: return mode2name[mode]
            return 'N/A'

        fig = plt.figure(figsize=(10, 10), dpi=600)
        ax = plt.subplot()
        censor_style = {'ms': 20, 'marker': '+'}
        
        temp = data[data['Grade']==0]
        kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade II")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=3, ls='--', markerfacecolor='black', censor_styles=censor_style)
        temp = data[data['grade_pred']==0]
        kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (Low)" % get_name(model))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=4, ls='-', markerfacecolor='black', censor_styles=censor_style)

        temp = data[data['Grade']==1]
        kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade III")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=3, ls='--', censor_styles=censor_style)
        temp = data[data['grade_pred']==1]
        kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (Mid)" % get_name(model))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=4, ls='-', censor_styles=censor_style)

        if subtype != 'ODG':    
            temp = data[data['Grade']==2]
            kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade IV")
            kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=3, ls='--', censor_styles=censor_style)
            temp = data[data['grade_pred']==2]
            kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (High)" % get_name(model))
            kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=4, ls='-', censor_styles=censor_style)

        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.001, 0.5))

        ax.tick_params(axis='both', which='major', labelsize=40)    
        plt.legend(fontsize=32, prop=font_manager.FontProperties(family='Arial', style='normal', size=32))
        if subtype != 'idhwt_ATC': ax.get_legend().remove()
        return fig
    
    data = poolSurvTestPD(ckpt_name, model, split, zscore, agg_type)
    for subtype in ['idhwt_ATC', 'idhmut_ATC', 'ODG']:
        fig = hazard2KMCurve(data[data['Histomolecular subtype'] == subtype], subtype)
        fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, subtype))
        
    fig = hazard2KMCurve(data, 'all')
    fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, 'all'))


def makeHazardSwarmPlot(ckpt_name='./checkpoints/surv_15_rnaseq/', model='path', split='test', zscore=True, agg_type='Hazard_mean'):
    mpl.rcParams['font.family'] = "arial"
    data = poolSurvTestPD(ckpt_name=ckpt_name, model=model, split=split, zscore=zscore, agg_type=agg_type)
    data = data[data['Grade'] != -1]
    data = data[data['Histomolecular subtype'] != -1]
    data['Grade'] = data['Grade'].astype(int).astype(str)
    data['Grade'] = data['Grade'].str.replace('0', 'Grade II', regex=False)
    data['Grade'] = data['Grade'].str.replace('1', 'Grade III', regex=False)
    data['Grade'] = data['Grade'].str.replace('2', 'Grade IV', regex=False)
    data['Histomolecular subtype'] = data['Histomolecular subtype'].str.replace('idhwt_ATC', 'IDH-wt \n astryocytoma', regex=False)
    data['Histomolecular subtype'] = data['Histomolecular subtype'].str.replace('idhmut_ATC', 'IDH-mut \n astrocytoma', regex=False)
    data['Histomolecular subtype'] = data['Histomolecular subtype'].str.replace('ODG', 'Oligodendroglioma', regex=False)

    fig, ax = plt.subplots(dpi=600)
    ax.set_ylim([-2, 2.5]) # plt.ylim(-2, 2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks(np.arange(-2, 2.001, 1))
    
    sns.swarmplot(x = 'Histomolecular subtype', y='Hazard', data=data, hue='Grade',
                  palette={"Grade II":"#AFD275" , "Grade III":"#7395AE", "Grade IV":"#E7717D"}, 
                  size = 4, alpha = 0.9, ax=ax)
    
    ax.set_xlabel('') # ax.set_xlabel('Histomolecular subtype', size=16)
    ax.set_ylabel('') # ax.set_ylabel('Hazard (Z-Score)', size=16)
    ax.tick_params(axis='y', which='both', labelsize=20)
    ax.tick_params(axis='x', which='both', labelsize=15)
    ax.tick_params(axis='x', which='both', labelbottom='off') # doesn't work??
    ax.legend(prop={'size': 8})
    fig.savefig(ckpt_name+'/%s_HSP.png' % (model))


def makeHazardBoxPlot(ckpt_name='./checkpoints/surv_15_rnaseq/', model='omic', split='test', zscore=True, agg_type='Hazard_mean'):
    mpl.rcParams['font.family'] = "arial"
    data = poolSurvTestPD(ckpt_name, model, split, zscore, 'Hazard_mean')
    data['Grade'] = data['Grade'].astype(int).astype(str)
    data['Grade'] = data['Grade'].str.replace('0', 'II', regex=False)
    data['Grade'] = data['Grade'].str.replace('1', 'III', regex=False)
    data['Grade'] = data['Grade'].str.replace('2', 'IV', regex=False)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [3, 3, 2]}, dpi=600)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.ylim(-2, 2)
    plt.yticks(np.arange(-2, 2.001, 1))
    #color_dict = {0: '#CF9498', 1: '#8CC7C8', 2: '#AAA0C6'}
    #color_dict = {0: '#F76C6C', 1: '#A8D0E6', 2: '#F8E9A1'}
    color_dict = ['#F76C6C', '#A8D0E6', '#F8E9A1']
    subtypes = ['idhwt_ATC', 'idhmut_ATC', 'ODG']

    for i in range(len(subtypes)):
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].xaxis.grid(False)
        axes[i].yaxis.grid(False)
        
        if i > 0: 
            axes[i].get_yaxis().set_visible(False)
            axes[i].spines["left"].set_visible(False)
            
        order = ["II","III","IV"] if subtypes[i] != 'ODG' else ["II", "III"]
        
        axes[i].xaxis.label.set_visible(False)
        axes[i].yaxis.label.set_visible(False)
        axes[i].tick_params(axis='y', which='both', labelsize=20)
        axes[i].tick_params(axis='x', which='both', labelsize=15)
        datapoints = data[data['Histomolecular subtype'] == subtypes[i]]
        sns.boxplot(y='Hazard', x="Grade", data=datapoints, ax = axes[i], color=color_dict[i], order=order)
        sns.stripplot(y='Hazard', x='Grade', data=datapoints, alpha=0.2, jitter=0.2, color='k', ax = axes[i], order=order)
        axes[i].set_ylim(-2.5, 2.5)
        axes[i].set_yticks(np.arange(-2.0, 2.1, 1))
        
    #axes[2].legend(prop={'size': 10})
    fig.savefig(ckpt_name+'/%s_HBP.png' % (model))


def makeAUROCPlot(ckpt_name='./checkpoints/grad_15/', model_list=['path', 'omic', 'pathgraphomic_fusion'], split='test', avg='micro', use_zoom=False):
    mpl.rcParams['font.family'] = "arial"
    colors = {'path':'dodgerblue', 'graph':'orange', 'omic':'green', 'pathgraphomic_fusion':'crimson'}
    names = {'path':'Histology CNN', 'graph':'Histology GCN', 'omic':'Genomic SNN', 'pathgraphomic_fusion':'Pathomic F.'}
    zoom_params = {0:([0.2, 0.4], [0.8, 1.0]), 
                   1:([0.25, 0.45], [0.75, 0.95]),
                   2:([0.0, 0.2], [0.8, 1.0]),
                   'micro':([0.15, 0.35], [0.8, 1.0])}
    mean_fpr = np.linspace(0, 1, 100)
    classes = [0, 1, 2, avg]
    ### 1. Looping over classes
    for i in classes:
        print("Class: " + str(i))
        fi = pylab.figure(figsize=(10,10), dpi=600, linewidth=0.2)
        axi = plt.subplot()
        
        ### 2. Looping over models
        for m, model in enumerate(model_list):
            ignore_missing_moltype = 1 if 'omic' in model else 0
            ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
            use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)

            ###. 3. Looping over all splits
            tprs, pres, aucrocs, rocaucs, = [], [], [], []
            for k in range(1,16):
                pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
                grade_pred, grade = np.array(pred[3]), np.array(pred[4])
                enc = LabelBinarizer()
                enc.fit(grade)
                grade_oh = enc.transform(grade)

                if i != avg:
                    pres.append(average_precision_score(grade_oh[:, i], grade_pred[:, i])) # from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
                    fpr, tpr, thresh = roc_curve(grade_oh[:,i], grade_pred[:,i], drop_intermediate=False)
                    aucrocs.append(auc(fpr, tpr)) # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
                    rocaucs.append(roc_auc_score(grade_oh[:,i], grade_pred[:,i])) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                else:
                    # A "micro-average": quantifying score on all classes jointly
                    pres.append(average_precision_score(grade_oh, grade_pred, average=avg))
                    fpr, tpr, thresh = roc_curve(grade_oh.ravel(), grade_pred.ravel())
                    aucrocs.append(auc(fpr, tpr))
                    rocaucs.append(roc_auc_score(grade_oh, grade_pred, avg))
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            #mean_auc = auc(mean_fpr, mean_tpr)
            mean_auc = np.mean(aucrocs)
            std_auc = np.std(aucrocs)
            print('\t'+'%s - AUC: %0.3f ± %0.3f' % (model, mean_auc, std_auc))
            
            if use_zoom:
                alpha, lw = (0.8, 6) if model =='pathgraphomic_fusion' else (0.5, 6)
                plt.plot(mean_fpr, mean_tpr, color=colors[model],
                     label=r'%s (AUC = %0.3f $\pm$ %0.3f)' % (names[model], mean_auc, std_auc), lw=lw, alpha=alpha)
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[model], alpha=0.1)
                plt.xlim([zoom_params[i][0][0]-0.005, zoom_params[i][0][1]+0.005])
                plt.ylim([zoom_params[i][1][0]-0.005, zoom_params[i][1][1]+0.005])
                axi.set_xticks(np.arange(zoom_params[i][0][0], zoom_params[i][0][1]+0.001, 0.05))
                axi.set_yticks(np.arange(zoom_params[i][1][0], zoom_params[i][1][1]+0.001, 0.05))
                axi.tick_params(axis='both', which='major', labelsize=26)
            else:
                alpha, lw = (0.8, 4) if model =='pathgraphomic_fusion' else (0.5, 3)
                plt.plot(mean_fpr, mean_tpr, color=colors[model],
                     label=r'%s (AUC = %0.3f $\pm$ %0.3f)' % (names[model], mean_auc, std_auc), lw=lw, alpha=alpha)
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[model], alpha=0.1)
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                axi.set_xticks(np.arange(0, 1.001, 0.2))
                axi.set_yticks(np.arange(0, 1.001, 0.2))
                axi.legend(loc="lower right", prop={'size': 20})
                axi.tick_params(axis='both', which='major', labelsize=30)
                #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=.8)

    figures = [manager.canvas.figure
               for manager in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
    
    zoom = '_zoom' if use_zoom else ''
    for i, fig in enumerate(figures):
        fig.savefig(ckpt_name+'/AUC_%s%s.png' % (classes[i], zoom))