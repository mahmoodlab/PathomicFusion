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

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


### Grade Classification
# Glioma
def getGradTestPats_GBMLGG(ckpt_name='./checkpoints/TCGA_GBMLGG/grad_15/', model='pathgraphomic_fusion', split='test', use_rnaseq=False, agg_type='mean'):
    pats = {}
    ignore_missing_moltype, ignore_missing_histype = 1, 1
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)
    use_rnaseq = '_rnaseq' if use_rnaseq else ''
    data_cv_path = '../data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq, )
    print(data_cv_path)
    for k in range(1,16):
        pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
        grad_all = pred[3].T
        grad_all = pd.DataFrame(np.stack(grad_all)).T
        grad_all.columns = ['score_0', 'score_1', 'score_2']
        data_cv = pickle.load(open(data_cv_path, 'rb'))
        data_cv_splits = data_cv['cv_splits']
        data_cv_split_k = data_cv_splits[k]
        assert np.all(data_cv_split_k[split]['g'] == pred[4]) # Data is correctly registered
        all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
        all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
        assert np.all(np.array(all_dataset_regstrd['Grade']) == pred[4])
        grad_all.index = data_cv_split_k[split]['x_patname']
        grad_all.index.name = 'TCGA ID'
        fun = p(0.75) if agg_type == 'p0.75' else agg_type
        grad_all = grad_all.groupby('TCGA ID').agg({'score_0': [fun], 'score_1': [fun], 'score_2': [fun]})
        pats[k] = grad_all.index
        
    return pats


def getPredAggGrad_GBMLGG(ckpt_name='./checkpoints/TCGA_GBMLGG/grad_15/', model='pathgraphomic_fusion', split='test', use_rnaseq=False, 
                         agg_type='max', test_pats=getGradTestPats_GBMLGG(), label='all'):
    y_label, y_pred = [], []

    ignore_missing_moltype = 1 if 'omic' in model else 0
    ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)
    use_rnaseq = '_rnaseq' if use_rnaseq else ''
    data_cv_path = '../data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq, )
    #print(data_cv_path)
    
    for k in range(1,16):
        ### Loads Prediction Pickle File. Registers predictions with TCGA IDs for the test split.
        pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
        grad_pred = pred[3].T
        grad_pred = pd.DataFrame(np.stack(grad_pred)).T
        grad_pred.columns = ['score_0', 'score_1', 'score_2']
        data_cv = pickle.load(open(data_cv_path, 'rb'))
        data_cv_splits = data_cv['cv_splits']
        data_cv_split_k = data_cv_splits[k]
        assert np.all(data_cv_split_k[split]['g'] == pred[4]) # Data is correctly registered
        all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
        all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
        assert np.all(np.array(all_dataset_regstrd['Grade']) == pred[4])
        grad_pred.index = data_cv_split_k[split]['x_patname']
        grad_pred.index.name = 'TCGA ID'
        
        ### Amalgamates predictions together.
        fun = p(0.90) if agg_type == 'p0.75' else agg_type
        grad_pred = grad_pred.groupby('TCGA ID').agg({'score_0': [fun], 'score_1': [fun], 'score_2': [fun]})
        test_pat = test_pats[k]
        grad_pred = grad_pred.loc[test_pat]
        grad_gt = np.array(all_dataset.loc[test_pat]['Grade'])
        grad_pred = np.array(grad_pred)
        enc = LabelBinarizer()
        enc.fit(grad_gt)
        grad_gt = enc.transform(grad_gt)
        
        y_label.append(grad_gt)
        y_pred.append(grad_pred)
    
    return y_label, y_pred

    y_label, y_pred = np.vstack(y_label), np.vstack(y_pred)
    if isinstance(label, int):
        return y_label[:,label], y_pred[:,label]
    return y_label, y_pred


def calcGradMetrics(y_label_all, y_pred_all, avg='micro'):
    rocauc_all = []
    ap_all = []
    f1_all = []
    f1_gradeIV_all = []
    
    for i in range(15):
        y_label, y_pred = y_label_all[i], y_pred_all[i]
        rocauc_all.append(roc_auc_score(y_label, y_pred, avg))

    return np.array(rocauc_all)


def calcAggGradMetrics(y_label_all, y_pred_all, avg='micro'):
    rocauc_all = []
    ap_all = []
    f1_all = []
    f1_gradeIV_all = []
    
    for i in range(15):
        y_label, y_pred = y_label_all[i], y_pred_all[i]
        rocauc_all.append(roc_auc_score(y_label, y_pred, avg))
        ap_all.append(average_precision_score(y_label, y_pred, average=avg))
        f1_all.append(f1_score(y_pred.argmax(axis=1), np.argmax(y_label, axis=1), average=avg))
        f1_gradeIV_all.append(f1_score(y_pred.argmax(axis=1), np.argmax(y_label, axis=1), average=None)[2])
        
    return np.array([CI_pm(rocauc_all), CI_pm(ap_all), CI_pm(f1_all), CI_pm(f1_gradeIV_all)])




def makeAUROCPlot(ckpt_name='./checkpoints/TCGA_GBMLGG/grad_15/', model_list=['path', 'pathgraphomic_fusion'], split='test', avg='micro', use_zoom=False):
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

            ###. 3. Looping over all splits
            tprs, pres, aucrocs, rocaucs, = [], [], [], []
            y_label_all, y_pred_all = getPredAggGrad_GBMLGG(model=model, agg_type='max')
            
            for k in range(15):
                y_label, y_pred = y_label_all[k], y_pred_all[k]
                
                if i != avg:
                    pres.append(average_precision_score(y_label[:, i], y_pred[:, i])) # from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
                    fpr, tpr, thresh = roc_curve(y_label[:,i], y_pred[:,i], drop_intermediate=False)
                    aucrocs.append(auc(fpr, tpr)) # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
                    rocaucs.append(roc_auc_score(y_label[:,i],y_pred[:,i])) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                else:
                    # A "micro-average": quantifying score on all classes jointly
                    pres.append(average_precision_score(y_label, y_pred, average=avg))
                    fpr, tpr, thresh = roc_curve(y_label.ravel(), y_pred.ravel())
                    aucrocs.append(auc(fpr, tpr))
                    rocaucs.append(roc_auc_score(y_label, y_pred, avg))
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
        fig.savefig(ckpt_name+'/AUC_%s%s.png' % (classes[i], zoom), bbox_inches='tight')

### Survival Outcome Prediction
def hazard2grade(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)


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


def p(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'p%s' % n
    return percentile_

def trainCox_GBMLGG(dataroot = './data/TCGA_GBMLGG/', ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='cox_omic', use_rnaseq=False, normalize=False, penalizer=0):
    ### Creates Checkpoint Directory
    if not os.path.exists(ckpt_name): os.makedirs(ckpt_name)
    if not os.path.exists(os.path.join(ckpt_name, model)): os.makedirs(os.path.join(ckpt_name, model))
    
    ### Load PNAS Splits
    pnas_splits = pd.read_csv(dataroot+'pnas_splits.csv')
    pnas_splits.columns = ['TCGA ID']+[str(k) for k in range(1, 16)]
    pnas_splits.index = pnas_splits['TCGA ID']
    pnas_splits = pnas_splits.drop(['TCGA ID'], axis=1)
    
    ### Loads Data
    ignore_missing_moltype = True if model in ['cox_omic', 'cox_moltype', 'cox_molgrade', 'all'] else False
    ignore_missing_histype = True if model in ['cox_histype', 'cox_grade', 'cox_molgrade', 'all'] else False
    all_dataset = getCleanGBMLGG(dataroot=dataroot, ignore_missing_moltype=ignore_missing_moltype, 
                                     ignore_missing_histype=ignore_missing_histype, use_rnaseq=use_rnaseq)[1]

    model_feats = {'cox_agegender':['Survival months', 'censored', 'Age', 'Gender'],
                   'cox_moltype':['Survival months', 'censored', 'codeletion', 'idh mutation'],
                   'cox_grade':['Survival months', 'censored', 'Grade'],
                   'cox_molgrade':['Survival months', 'censored', 'Grade', 'codeletion', 'idh mutation'],
                   'cox_covariates':['Survival months', 'censored', 'codeletion', 'idh mutation', 'Grade', 'Age', 'Gender', 'Histology']}
    cv_results = []
    cv_pvals = []

    for k in pnas_splits.columns:
        pat_train = list(set(pnas_splits.index[pnas_splits[k] == 'Train']).intersection(all_dataset.index))
        pat_test = list(set(pnas_splits.index[pnas_splits[k] == 'Test']).intersection(all_dataset.index))
        feats = all_dataset.columns.drop(model_feats[model]) if model == 'cox_omic' or model == 'cox_all' else model_feats[model]
        train = all_dataset.loc[pat_train]
        test = all_dataset.loc[pat_test]
        
        if normalize:
            scaler = preprocessing.StandardScaler().fit(train[feats])
            train[feats] = scaler.transform(train[feats])
            test[feats] = scaler.transform(test[feats])

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(train[feats], duration_col='Survival months', event_col='censored', show_progress=False)
        cin = concordance_index(test['Survival months'], -cph.predict_partial_hazard(test[feats]), test['censored'])
        pval = cox_log_rank(np.array(-cph.predict_partial_hazard(test[feats])).reshape(-1), 
                            np.array(test['censored']).reshape(-1), 
                            np.array(test['Survival months']).reshape(-1))
        cv_results.append(cin)
        cv_pvals.append(pval)
        
        train.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(train))
        test.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(test))
        pickle.dump(train, open(os.path.join(ckpt_name, model, '%s_%s_pred_train.pkl' % (model, k)), 'wb'))
        pickle.dump(test, open(os.path.join(ckpt_name, model, '%s_%s_pred_test.pkl' % (model, k)), 'wb'))
        
    pickle.dump(cv_results, open(os.path.join(ckpt_name, model, '%s_results.pkl' % model), 'wb'))
    print("C-Indices across Splits", cv_results)
    print("Average C-Index: %s" % CI_pm(cv_results))


def getSurvTestPats_GBMLGG(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', split='test', use_rnaseq=True, agg_type='Hazard_mean'):
    pats = {}

    print(model)
    ignore_missing_moltype = 1
    ignore_missing_histype = 0
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)
    use_rnaseq = '_rnaseq' if use_rnaseq else ''
    data_cv_path = './data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq, )
    print(data_cv_path)
    
    for k in range(1,16):
        pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
        surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
        surv_all.columns = ['Hazard', 'Survival months', 'censored', 'Grade']
        data_cv = pickle.load(open(data_cv_path, 'rb'))
        data_cv_splits = data_cv['cv_splits']
        data_cv_split_k = data_cv_splits[k]
        assert np.all(data_cv_split_k[split]['t'] == pred[1]) # Data is correctly registered
        all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
        all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
        assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
        assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
        assert np.all(np.array(all_dataset_regstrd['Grade']) == pred[4])
        all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))
        
        hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
        hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
        hazard_agg = hazard_agg[[agg_type]]
        hazard_agg.columns = ['Hazard']
        all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')
        pats[k] = all_dataset_hazard.index
        
    return pats


def getPValAggSurv_GBMLGG_Binary(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', percentile=[50]):
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model)
    p = np.percentile(data['Hazard'], percentile)
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_high = data['Survival months'][data['grade_pred']==0], data['Survival months'][data['grade_pred']==1]
    E_low, E_high = data['censored'][data['grade_pred']==0], data['censored'][data['grade_pred']==1]

    low_vs_high = logrank_test(durations_A=T_low, durations_B=T_high, event_observed_A=E_low, event_observed_B=E_high).p_value
    return np.array([low_vs_high])


def getPValAggSurv_GBMLGG_Multi(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', percentile=[33,66]):
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model)
    p = np.percentile(data['Hazard'], percentile)
    if p[0] == p[1]: p[0] = 2.99997
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_mid, T_high = data['Survival months'][data['grade_pred']==0], data['Survival months'][data['grade_pred']==1], data['Survival months'][data['grade_pred']==2]
    E_low, E_mid, E_high = data['censored'][data['grade_pred']==0], data['censored'][data['grade_pred']==1], data['censored'][data['grade_pred']==2]
    low_vs_mid = logrank_test(durations_A=T_low, durations_B=T_mid, event_observed_A=E_low, event_observed_B=E_mid).p_value
    mid_vs_high = logrank_test(durations_A=T_mid, durations_B=T_high, event_observed_A=E_mid, event_observed_B=E_high).p_value
    return np.array([low_vs_mid, mid_vs_high])


def getPredAggSurv_GBMLGG(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', 
                       split='test', use_rnaseq=True, agg_type='Hazard_mean', test_pats=getSurvTestPats_GBMLGG()):
    results = []
    if 'cox' in model:
        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d_pred_%s.pkl' % (model, model, k, split), 'rb'))    
            cin = CIndex_lifeline(-pred['Hazard'], pred['censored'], pred['Survival months'])
            results.append(cin)
        return results
    else:
        ignore_missing_moltype = 1 if 'omic' in model else 0
        ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
        use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)
        use_rnaseq = '_rnaseq' if ('rnaseq' in ckpt_name and 'path' != model and 'pathpath' not in model and 'graph' != model and 'graphgraph' not in model) else ''
        data_cv_path = './data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq, )
        data_cv = pickle.load(open(data_cv_path, 'rb'))

        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
            surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
            surv_all.columns = ['Hazard', 'Survival months', 'censored', 'Grade']
            data_cv_splits = data_cv['cv_splits']
            data_cv_split_k = data_cv_splits[k]
            assert np.all(data_cv_split_k[split]['t'] == pred[1]) # Data is correctly registered
            all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
            all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
            assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
            assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
            assert np.all(np.array(all_dataset_regstrd['Grade']) == pred[4])
            all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))

            hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
            hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
            hazard_agg = hazard_agg[[agg_type]]
            hazard_agg.columns = ['Hazard']
            all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')

            all_dataset_hazard = all_dataset_hazard.loc[test_pats[k]]

            cin = CIndex_lifeline(all_dataset_hazard['Hazard'], all_dataset_hazard['censored'], all_dataset_hazard['Survival months'])
            results.append(cin)

        return results


def getDataAggSurv_GBMLGG(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', 
                       split='test', use_rnaseq=True, agg_type='Hazard_mean', zscore=False, test_pats=getSurvTestPats_GBMLGG()):
    data = []
    
    if 'cox' in model:
        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d_pred_%s.pkl' % (model, model, k, split), 'rb'))    
            data.append(pred)
        return pd.concat(data)
    else:
        ignore_missing_moltype = 1 if 'omic' in model else 0
        ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
        use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)
        use_rnaseq = '_rnaseq' if ('rnaseq' in ckpt_name and 'path' != model and 'pathpath' not in model and 'graph' != model and 'graphgraph' not in model) else ''
        data_cv_path = './data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq, )
        data_cv = pickle.load(open(data_cv_path, 'rb'))

        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
            surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
            surv_all.columns = ['Hazard', 'Survival months', 'censored', 'Grade']
            data_cv_splits = data_cv['cv_splits']
            data_cv_split_k = data_cv_splits[k]
            assert np.all(data_cv_split_k[split]['t'] == pred[1]) # Data is correctly registered
            all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
            all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
            assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
            assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
            assert np.all(np.array(all_dataset_regstrd['Grade']) == pred[4])
            all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))

            hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
            hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
            hazard_agg = hazard_agg[[agg_type]]
            hazard_agg.columns = ['Hazard']
            all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')
            all_dataset_hazard = all_dataset_hazard.loc[test_pats[k]]
            all_dataset_hazard['split'] = k
            if zscore: all_dataset_hazard['Hazard'] = scipy.stats.zscore(np.array(all_dataset_hazard['Hazard']))

            data.append(all_dataset_hazard)

        data = pd.concat(data)
        data = changeHistomolecularSubtype(data)
        return data

def getHazardHistogramPlot_GBMLGG(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', 
                           split='test', zscore=True, agg_type='Hazard_mean', c=[(-1.5, -0.5), (1, 1.25), (1.25, 1.5)]):
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model, split=split, use_rnaseq=True, agg_type=agg_type, zscore=zscore)
    norm = True
    fig, ax = plt.subplots(dpi=600)

    low = data[data['Survival months'] <= 365*5]
    low = low[low['censored'] == 1]
    high = data[data['Survival months'] > 365*5]
    high = high[high['censored'] == 1]

    sns.distplot(low['Hazard'], bins=15, kde=False, norm_hist=norm,
                 #kde_kws={"color": "k", "lw": 2},
                 hist_kws={'histtype':'stepfilled', "linewidth": 1, "alpha": 0.5, "color": "r"}, ax=ax)
    sns.distplot(high['Hazard'], bins=15, kde=False, norm_hist=norm,
                 #kde_kws={"color": "k", "lw": 2},
                 hist_kws={'histtype':'stepfilled', "linewidth": 1, "alpha": 0.5, "color": "b"}, ax=ax)

    ax.set_xlabel('')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='y', which='both', labelsize=15)
    ax.tick_params(axis='x', which='both', labelsize=15)
    ax.set_xticks(np.arange(-1.5, 1.51, 0.5))
    plt.xlim([-1.75, 1.75])
    if norm:
        ax.set_yticks(np.arange(0, 2.1, 1))
        plt.ylim([0, 2])

    fig.savefig(ckpt_name+'/%s_HHP_V2.png' % (model))

    cluster1 = data[data['Hazard'] > c[0][0]]
    cluster1 = cluster1[cluster1['Hazard'] < c[0][1]]
    num_cluster1 = cluster1.shape[0]
    cluster1_II = (cluster1['Grade'] == 0).sum() / num_cluster1
    cluster1_III = (cluster1['Grade'] == 1).sum() / num_cluster1
    cluster1_IV = (cluster1['Grade'] == 2).sum() / num_cluster1
    cluster1_ODG = (cluster1['Histomolecular subtype'] == 'ODG').sum() / num_cluster1
    cluster1_IDHmut = (cluster1['Histomolecular subtype'] == 'idhmut_ATC').sum() / num_cluster1
    cluster1_IDHwt = (cluster1['Histomolecular subtype'] == 'idhwt_ATC').sum() / num_cluster1
    cluster1_summary = [cluster1_II, cluster1_III, cluster1_IV, cluster1_ODG, cluster1_IDHmut, cluster1_IDHwt]

    cluster2 = data[data['Hazard'] > c[1][0]]
    cluster2 = cluster2[cluster2['Hazard'] < c[1][1]]
    num_cluster2 = cluster2.shape[0]
    cluster2_II = (cluster2['Grade'] == 0).sum() / num_cluster2
    cluster2_III = (cluster2['Grade'] == 1).sum() / num_cluster2
    cluster2_IV = (cluster2['Grade'] == 2).sum() / num_cluster2
    cluster2_ODG = (cluster2['Histomolecular subtype'] == 'ODG').sum() / num_cluster2
    cluster2_IDHmut = (cluster2['Histomolecular subtype'] == 'idhmut_ATC').sum() / num_cluster2
    cluster2_IDHwt = (cluster2['Histomolecular subtype'] == 'idhwt_ATC').sum() / num_cluster2
    cluster2_summary = [cluster2_II, cluster2_III, cluster2_IV, cluster2_ODG, cluster2_IDHmut, cluster2_IDHwt]

    cluster3 = data[data['Hazard'] > c[2][0]]
    cluster3 = cluster3[cluster3['Hazard'] < c[2][1]]
    num_cluster3 = cluster3.shape[0]
    cluster3_II = (cluster3['Grade'] == 0).sum() / num_cluster3
    cluster3_III = (cluster3['Grade'] == 1).sum() / num_cluster3
    cluster3_IV = (cluster3['Grade'] == 2).sum() / num_cluster3
    cluster3_ODG = (cluster3['Histomolecular subtype'] == 'ODG').sum() / num_cluster3
    cluster3_IDHmut = (cluster3['Histomolecular subtype'] == 'idhmut_ATC').sum() / num_cluster3
    cluster3_IDHwt = (cluster3['Histomolecular subtype'] == 'idhwt_ATC').sum() / num_cluster3
    cluster3_summary = [cluster3_II, cluster3_III, cluster3_IV, cluster3_ODG, cluster3_IDHmut, cluster3_IDHwt]

    cluster_results = pd.DataFrame([cluster1_summary, cluster2_summary, cluster3_summary])
    cluster_results.index = ['%0.2f < Hazard < %0.2f' % c[0], '%0.2f < Hazard < %0.2f'  % c[1], '%0.2f < Hazard < %0.2f' % c[2]]
    cluster_results.index.name = 'Density Region'
    cluster_results.columns = ['Grade II (%)', 'Grade III (%)', 'Grade IV (%)', 'ODG (%)', 'IDHmut ATC (%)', 'IDHwt ATC (%)']
    cluster_results *= 100
    pd.options.display.float_format = '{:.2f}'.format
    return cluster_results

def makeHazardSwarmPlot(ckpt_name='./checkpoints/surv_15_rnaseq/', model='path', split='test', zscore=True, agg_type='Hazard_mean'):
    mpl.rcParams['font.family'] = "arial"
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model, split=split, zscore=zscore, agg_type=agg_type)
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
    plt.close()


def makeHazardBoxPlot(ckpt_name='./checkpoints/surv_15_rnaseq/', model='omic', split='test', zscore=True, agg_type='Hazard_mean'):
    mpl.rcParams['font.family'] = "arial"
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model, split=split, zscore=zscore, agg_type=agg_type)
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
    plt.close()


def makeKaplanMeierPlot(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='omic', split='test', zscore=False, agg_type='Hazard_mean', plot_gt=True):
    def hazard2KMCurve(data, subtype, plot_gt=True):
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
        if plot_gt:
            temp = data[data['Grade']==0]
            kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade II")
            kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=3, ls='--', markerfacecolor='black', censor_styles=censor_style)
        temp = data[data['grade_pred']==0]
        kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (Low)" % get_name(model))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=4, ls='-', markerfacecolor='black', censor_styles=censor_style)

        if plot_gt:
            temp = data[data['Grade']==1]
            kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade III")
            kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=3, ls='--', censor_styles=censor_style)
        temp = data[data['grade_pred']==1]
        kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (Int.)" % get_name(model))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=4, ls='-', censor_styles=censor_style)

        if subtype != 'ODG':
            if plot_gt:
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
        plt.legend(fontsize=24, prop=font_manager.FontProperties(family='Arial', style='normal', size=24))
        if subtype != 'idhwt_ATC' and plot_gt: ax.get_legend().remove()
        return fig
    
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model)

    for subtype in ['idhwt_ATC', 'idhmut_ATC', 'ODG']:
        if plot_gt:
            fig = hazard2KMCurve(data[data['Histomolecular subtype'] == subtype], subtype)
            fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, subtype), bbox_inches='tight')
            plt.close()

    fig = hazard2KMCurve(data, 'all', plot_gt=plot_gt)
    fig.savefig(ckpt_name+'/%s_KM_%s%s.png' % (model, 'all', '' if plot_gt else '_nogt'), bbox_inches='tight')
    plt.close()

def makeKaplanMeierPlot_Baseline(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='Grade'):
    def hazard2KMCurve(data, model):
        fig = plt.figure(figsize=(10, 10), dpi=600)
        ax = plt.subplot()
        censor_style = {'ms': 20, 'marker': '+'}
        kmf = lifelines.KaplanMeierFitter()
        kmf_gt = lifelines.KaplanMeierFitter()
        
        baseline = {'Grade':[0,1,2], 
                    'Histomolecular subtype':['ODG', 'idhmut_ATC', 'idhwt_ATC']}
        baseline_name = {'Grade':['Grade II', 'Grade III', 'Grade IV'], 
                         'Histomolecular subtype':['Oligodendroglioma', 'IDHmut Astrocytoma', 'IDHwt Astrocytoma']}
        
        temp = data[data['Grade']==0]
        kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade II")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=3, ls='--', markerfacecolor='black', censor_styles=censor_style)
        temp = data[data[model]==baseline[model][0]]
        kmf.fit(temp['Survival months']/365, temp['censored'], label=baseline_name[model][0])
        kmf.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=4, ls='-', markerfacecolor='black', censor_styles=censor_style)

        temp = data[data['Grade']==1]
        kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade III")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=3, ls='--', censor_styles=censor_style)
        temp = data[data[model]==baseline[model][1]]
        kmf.fit(temp['Survival months']/365, temp['censored'], label=baseline_name[model][1])
        kmf.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=4, ls='-', censor_styles=censor_style)
        
        temp = data[data['Grade']==2]
        kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade IV")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=3, ls='--', censor_styles=censor_style)
        temp = data[data[model]==baseline[model][2]]
        kmf.fit(temp['Survival months']/365, temp['censored'], label=baseline_name[model][2])
        kmf.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=4, ls='-', censor_styles=censor_style)
            
        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.001, 0.5))

        ax.tick_params(axis='both', which='major', labelsize=40)    
        plt.legend(fontsize=24, prop=font_manager.FontProperties(family='Arial', style='normal', size=24))
        return fig
    
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model='pathgraphomic_fusion')

    fig = hazard2KMCurve(data, model)
    fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, 'all'), bbox_inches='tight')


### KIRC
def getDataAggSurv_KIRC(ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='pathgraphomic_fusion', 
                       split='test', use_rnaseq=True, agg_type='Hazard_mean', zscore=False):
    data = []
    if 'cox' in model:
        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d_pred_%s.pkl' % (model, model, k, split), 'rb'))    
            data.append(pred)
        return pd.concat(data)
    else:
        data_cv = pickle.load(open('./data/TCGA_KIRC/splits/KIRC_st_1.pkl', 'rb'))
        data_cv_splits = data_cv['split']

        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d_pred_%s.pkl' % (model, model, k, split), 'rb'))    
            surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
            surv_all.columns = ['Hazard', 'OS_month', 'censored', 'Grade']
            data_cv_split_k = data_cv_splits[k]
            assert np.all(data_cv_split_k[split]['t'] == pred[1]) # Data is correctly registered
            all_dataset = data_cv['all_dataset']
            all_dataset.index = all_dataset.index.str[:12]
            
            patnames = data_cv_split_k[split]['x_patname']
            patnames = [pat[:12] for pat in patnames]

            all_dataset_regstrd = all_dataset.loc[patnames] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
            assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
            all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))
            all_dataset_regstrd['TCGA ID'] = all_dataset_regstrd.index

            hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
            hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
            hazard_agg = hazard_agg[[agg_type]]
            hazard_agg.columns = ['Hazard']
            all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')
            all_dataset_hazard['split'] = k
            if zscore: all_dataset_hazard['Hazard'] = scipy.stats.zscore(np.array(all_dataset_hazard['Hazard']))
            data.append(all_dataset_hazard)

        data = addNeoplasmGrade(pd.concat(data))
        return data


def getPredAggSurv_KIRC(ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='pathgraphomic_fusion', 
                       split='test', use_rnaseq=True, agg_type='Hazard_mean', test_pats=None):
    results = []
    if 'cox' in model:
        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d_pred_%s.pkl' % (model, model, k, split), 'rb'))    
            cin = CIndex_lifeline(-pred['Hazard'], pred['censored'], pred['OS_month'])
            results.append(cin)
        return results
    else:
        data_cv = pickle.load(open('./data/TCGA_KIRC/splits/KIRC_st_1.pkl', 'rb'))
        data_cv_splits = data_cv['split']

        for k in range(1,16):
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d_pred_%s.pkl' % (model, model, k, split), 'rb'))    
            surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
            surv_all.columns = ['Hazard', 'OS_month', 'censored', 'Grade']
            data_cv_split_k = data_cv_splits[k]
            assert np.all(data_cv_split_k[split]['t'] == pred[1]) # Data is correctly registered
            all_dataset = data_cv['all_dataset']
            #return all_dataset
            
            patnames = data_cv_split_k[split]['x_patname']
            patnames = [pat[:12] for pat in patnames]

            all_dataset_regstrd = all_dataset.loc[patnames] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
            assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
            all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))

            all_dataset_regstrd['TCGA ID'] = all_dataset_regstrd.index

            hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
            hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
            hazard_agg = hazard_agg[[agg_type]]
            hazard_agg.columns = ['Hazard']
            all_dataset_hazard = addNeoplasmGrade(hazard_agg.join(all_dataset, how='inner'))   
            cin = CIndex_lifeline(all_dataset_hazard['Hazard'], all_dataset_hazard['censored'], all_dataset_hazard['OS_month'])
            results.append(cin)

        return results

def hazard2grade(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)


def getPValAggSurv_KIRC_Binary(ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='pathgraphomic_fusion', percentile=[50]):
    data = getDataAggSurv_KIRC(ckpt_name=ckpt_name, model=model)
    p = np.percentile(data['Hazard'], percentile)
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_high = data['OS_month'][data['grade_pred']==0], data['OS_month'][data['grade_pred']==1]
    E_low, E_high = data['censored'][data['grade_pred']==0], data['censored'][data['grade_pred']==1]

    low_vs_high = logrank_test(durations_A=T_low, durations_B=T_high, event_observed_A=E_low, event_observed_B=E_high).p_value
    return np.array([low_vs_high])


def getPValAggSurv_KIRC_Multi(ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='pathgraphomic_fusion', percentile=[26,51,76]):
    data = getDataAggSurv_KIRC(ckpt_name=ckpt_name, model=model)
    p = np.percentile(data['Hazard'], percentile)
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_midlow = data['OS_month'][data['grade_pred']==0], data['OS_month'][data['grade_pred']==1]
    T_midhigh, T_high = data['OS_month'][data['grade_pred']==2], data['OS_month'][data['grade_pred']==3]
    E_low, E_midlow = data['censored'][data['grade_pred']==0], data['censored'][data['grade_pred']==1]
    E_midhigh, E_high = data['censored'][data['grade_pred']==2], data['censored'][data['grade_pred']==3]

    low_vs_midlow = logrank_test(durations_A=T_low, durations_B=T_midlow, event_observed_A=E_low, event_observed_B=E_midlow).p_value
    midlow_vs_midhigh = logrank_test(durations_A=T_midlow, durations_B=T_midhigh, event_observed_A=E_midlow, event_observed_B=E_midhigh).p_value
    midhigh_vs_high = logrank_test(durations_A=T_midhigh, durations_B=T_high, event_observed_A=E_midhigh, event_observed_B=E_high).p_value
    return np.array([low_vs_midlow, midlow_vs_midhigh, midhigh_vs_high])


def addNeoplasmGrade(data):
    clinical_with_grade = pd.read_csv('./data/TCGA_KIRC/kirc_tcga_pan_can_atlas_2018_clinical_data.tsv', sep='\t', index_col=1)
    clinical_with_grade.index.name = None
    data = data.join(clinical_with_grade[['Neoplasm Histologic Grade']], how='inner')
    data['Neoplasm Histologic Grade'] = data['Neoplasm Histologic Grade'].str.lstrip('G')
    data = data[~data['Neoplasm Histologic Grade'].isnull()]
    data = data[data['Neoplasm Histologic Grade'] != 'X']
    return data



def CI_pm(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return str("{0:.4f} ± ".format(m) + "{0:.3f}".format(h))


def trainCox_KIRC(dataroot = './data/TCGA_KIRC/', ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='cox_agegender', normalize=False, penalizer=1e-4):
    ### Creates Checkpoint Directory
    if not os.path.exists(ckpt_name): os.makedirs(ckpt_name)
    if not os.path.exists(os.path.join(ckpt_name, model)): os.makedirs(os.path.join(ckpt_name, model))
    
    clinical = pd.read_table(os.path.join(dataroot, './kirc_tcga_pan_can_atlas_2018_clinical_data.tsv'), index_col=2)
    clinical.index.name = None
    clinical = clinical[['Center of sequencing', 'Overall Survival Status', 'Overall Survival (Months)', 'Diagnosis Age', 'Sex', 'Neoplasm Histologic Grade']].copy()
    clinical = clinical.rename(columns={'Center of sequencing':'CoS', 'Overall Survival Status':'censored', 'Overall Survival (Months)':'OS_month', 'Diagnosis Age':'Age', 'Neoplasm Histologic Grade':'Grade'})
    clinical['Sex'] = clinical['Sex'].replace({'Male':0, 'Female': 1})
    clinical['censored'] = clinical['censored'].replace('LIVING', 0) # actually uncensored
    clinical['censored'] = clinical['censored'].replace('DECEASED', 1) # actually uncensored
    clinical['train'] = 0
    train_cohort = list(clinical['CoS'].value_counts().index[0:2]) + list(clinical['CoS'].value_counts().index[-16:])
    clinical.loc[clinical['CoS'].isin(train_cohort), 'train'] = 1
    clinical = clinical.sort_values(['train', 'CoS'], ascending=False)
    clinical['Grade'] = clinical['Grade'].str.lstrip('G')
    clinical = clinical[~clinical['Grade'].isnull()]
    clinical = clinical[clinical['Grade'] != 'X']
    clinical = clinical.drop(['CoS'], axis=1)
    all_dataset = clinical
    all_dataset.index = all_dataset.index.str[:-3]
    
    model_feats = {'cox_agegender':['OS_month', 'censored', 'Age', 'Sex'],
                   'cox_grade':['OS_month', 'censored', 'Grade'],
                   'cox_all':['OS_month', 'censored', 'Age', 'Sex', 'Grade']}
    cv_results = []
    cv_pvals = []

    splits = pd.read_csv(os.path.join(dataroot, 'kirc_splits.csv'), index_col=0)
    splits.columns = [str(k) for k in range(1, 16)]

    for k in range(1,16):
        feats = model_feats[model]
        pat_train = splits.index[splits[str(k)] == 'Train']
        pat_test = splits.index[splits[str(k)] == 'Test']
        train = all_dataset.loc[pat_train.intersection(all_dataset.index)]
        test = all_dataset.loc[pat_test.intersection(all_dataset.index)]
        
        if normalize:
            scaler = preprocessing.StandardScaler().fit(train[feats])
            train[feats] = scaler.transform(train[feats])
            test[feats] = scaler.transform(test[feats])

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(train[feats], duration_col='OS_month', event_col='censored', show_progress=False)
        cin = concordance_index(test['OS_month'], -cph.predict_partial_hazard(test[feats]), test['censored'])
        pval = cox_log_rank(np.array(-cph.predict_partial_hazard(test[feats])).reshape(-1), 
                            np.array(test['censored']).reshape(-1), 
                            np.array(test['OS_month']).reshape(-1))
        cv_results.append(cin)
        cv_pvals.append(pval)
        
        train.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(train))
        test.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(test))
        pickle.dump(train, open(os.path.join(ckpt_name, model, '%s_%s_pred_train.pkl' % (model, k)), 'wb'))
        pickle.dump(test, open(os.path.join(ckpt_name, model, '%s_%s_pred_test.pkl' % (model, k)), 'wb'))
        
    pickle.dump(cv_results, open(os.path.join(ckpt_name, model, '%s_results.pkl' % model), 'wb'))
    print("C-Indices across Splits", cv_results)
    print("Average C-Index: %s" % CI_pm(cv_results))
    print("Average P-Value: " + str(np.mean(cv_pvals)))


def getHazardHistogramPlot_KIRC(ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='pathgraphomic_fusion', 
                           split='test', zscore=True, agg_type='Hazard_mean', c=[(-1.5, -0.75), (-0.75, 0), (0, 0.75), (0.75, 1.5)]):
    data = getDataAggSurv_KIRC(ckpt_name=ckpt_name, model=model, split=split, use_rnaseq=True, agg_type=agg_type, zscore=zscore)
    norm = True
    fig, ax = plt.subplots(dpi=600)

    low = data[data['OS_month'] <= 12*3.5]
    low = low[low['censored'] == 1]
    high = data[data['OS_month'] > 12*3.5]
    high = high[high['censored'] == 1]

    sns.distplot(low['Hazard'], bins=15, kde=False, norm_hist=norm,
                 #kde_kws={"color": "k", "lw": 2},
                 hist_kws={'histtype':'stepfilled', "linewidth": 1, "alpha": 0.5, "color": "r"}, ax=ax)
    sns.distplot(high['Hazard'], bins=15, kde=False, norm_hist=norm,
                 #kde_kws={"color": "k", "lw": 2},
                 hist_kws={'histtype':'stepfilled', "linewidth": 1, "alpha": 0.5, "color": "b"}, ax=ax)

    ax.set_xlabel('')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='y', which='both', labelsize=15)
    ax.tick_params(axis='x', which='both', labelsize=15)
    ax.set_xticks(np.arange(-1.5, 1.51, 0.5))
    plt.xlim([-1.75, 1.75])
    if norm:
        ax.set_yticks(np.arange(0, 2.1, 1))
        plt.ylim([0, 2])

    fig.savefig(ckpt_name+'/%s_HHP.png' % (model))
    
    clusters = []
    for i in range(len(c)):
        cluster = data[data['Hazard'] > c[i][0]]
        cluster = cluster[cluster['Hazard'] < c[i][1]]
        cluster_size = cluster.shape[0]
        cluster_I = (cluster['Neoplasm Histologic Grade'] == '1').sum() / cluster_size
        cluster_II = (cluster['Neoplasm Histologic Grade'] == '2').sum() / cluster_size
        cluster_III = (cluster['Neoplasm Histologic Grade'] == '3').sum() / cluster_size
        cluster_IV = (cluster['Neoplasm Histologic Grade'] == '4').sum() / cluster_size
        cluster_summary = [cluster_I, cluster_II, cluster_III, cluster_IV]
        clusters.append(cluster_summary)

    cluster_results = pd.DataFrame(clusters)
    cluster_results.index = ['%0.2f < Hazard < %0.2f' % c[i] for i in range(len(c))]
    cluster_results.index.name = 'Density Region'
    cluster_results.columns = ['Grade ' + str(g) + ' (%)' for g in range(1, 5)]
    cluster_results *= 100
    pd.options.display.float_format = '{:.2f}'.format
    return cluster_results


def hazard2grade(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)

def makeKaplanMeierPlot_KIRC_Binary(ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='omic', split='test', zscore=False, agg_type='Hazard_mean', percentile=[50]):
    def hazard2KMCurve(data, percentile):
        p = np.percentile(data['Hazard'], percentile)
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
        
        for i in range(len(p)):
            temp = data[data['grade_pred']==i]
            kmf_pred.fit(temp['OS_month']/12, temp['censored'], label="%s (Low)" % get_name(model))
            kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=4, ls='-', markerfacecolor='black', censor_styles=censor_style)

        temp = data[data['grade_pred']==len(p)]
        kmf_pred.fit(temp['OS_month']/12, temp['censored'], label="%s (High)" % get_name(model))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=4, ls='-', censor_styles=censor_style)
 
        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.001, 0.5))

        ax.tick_params(axis='both', which='major', labelsize=40)    
        plt.legend(fontsize=32, prop=font_manager.FontProperties(family='Arial', style='normal', size=32))
            
        return fig
    
    data = getDataAggSurv_KIRC(ckpt_name=ckpt_name, model=model)

    fig = hazard2KMCurve(data, percentile=percentile)
    fig.savefig(ckpt_name+'/%s_%s_KM_%s.png' % ('./checkpoints/TCGA_KIRC/surv_15/'.split('/')[2], model, 'all'), bbox_inches='tight')
    plt.close()
    

def makeKaplanMeierPlot_KIRC_Multi(ckpt_name='./checkpoints/TCGA_KIRC/surv_15/', model='omic', split='test', zscore=False, agg_type='Hazard_mean', percentile=[25, 50, 75]):
    def hazard2KMCurve(data, percentile):
        p = np.percentile(data['Hazard'], percentile)
        print(p)
        return None
        data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
        kmf_pred = lifelines.KaplanMeierFitter()
        kmf_gt = lifelines.KaplanMeierFitter()

        def get_name(model):
            mode2name = {'pathgraphomic':'Pathomic F.', 'pathomic':'Pathomic F.', 'graphomic':'Pathomic F.', 'path':'Hist. CNN', 'graph':'Histology GCN', 'omic':'Genomic SNN'}
            for mode in mode2name.keys():
                if mode in model: return mode2name[mode]
            return 'N/A'

        fig = plt.figure(figsize=(10, 10), dpi=600)
        ax = plt.subplot()
        censor_style = {'ms': 20, 'marker': '+'}
        
        colors = ['g', 'b', 'm', 'r']
        stage = ['G1', 'G2', 'G3', 'G4']
        pred = ['<25%', '25-50%', '50-75%', '>75%']
        
        for i in range(len(p)):
            temp = data[data['Neoplasm Histologic Grade']==str(i+1)]
            kmf_gt.fit(temp['OS_month']/12, temp['censored'], label=stage[i])
            kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c=colors[i], linewidth=3, ls='--', markerfacecolor='black', censor_styles=censor_style)
            temp = data[data['grade_pred']==i]
            kmf_pred.fit(temp['OS_month']/12, temp['censored'], label="%s (%s)" % (get_name(model), pred[i]))
            kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c=colors[i], linewidth=4, ls='-', markerfacecolor='black', censor_styles=censor_style)

        temp = data[data['Neoplasm Histologic Grade']==str(4)]
        kmf_gt.fit(temp['OS_month']/12, temp['censored'], label=stage[i])
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=3, ls='--', markerfacecolor='black', censor_styles=censor_style)
        temp = data[data['grade_pred']==len(p)]
        kmf_pred.fit(temp['OS_month']/12, temp['censored'], label="%s (%s)" % (get_name(model), pred[i]))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=4, ls='-', censor_styles=censor_style)
 
        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.001, 0.5))

        ax.tick_params(axis='both', which='major', labelsize=40)    
        plt.legend(fontsize=16, prop=font_manager.FontProperties(family='Arial', style='normal', size=16))
        #ax.get_legend().remove()
        return fig
    
    data = getDataAggSurv_KIRC(ckpt_name=ckpt_name, model=model)

    fig = hazard2KMCurve(data, percentile=percentile)
    return None
    fig.savefig(ckpt_name+'/%s_%s_KM_%s.png' % ('./checkpoints/TCGA_KIRC/surv_15/'.split('/')[2]+"_Multi", model, 'all'), bbox_inches='tight')
    plt.close()
