# Base / Native
import os
import pickle

# Numerical / Array
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
pd.options.display.max_rows = 999

# Env
from utils import CI_pm
from utils import cox_log_rank
from utils import getCleanAllDataset, addHistomolecularSubtype
from utils import makeKaplanMeierPlot



def trainCox(dataroot = './data/TCGA_GBMLGG/', ckpt_name='./checkpoints/surv_15_cox/', model='cox_omic', penalizer=1e-4):
    ### Creates Checkpoint Directory
    if not os.path.exists(ckpt_name): os.makedirs(ckpt_name)
    if not os.path.exists(os.path.join(ckpt_name, model)): os.makedirs(os.path.join(ckpt_name, model))
    
    ### Load PNAS Splits
    pnas_splits = pd.read_csv(dataroot+'pnas_splits.csv')
    pnas_splits.columns = ['TCGA ID']+[str(k) for k in range(1, 16)]
    pnas_splits.index = pnas_splits['TCGA ID']
    pnas_splits = pnas_splits.drop(['TCGA ID'], axis=1)
    
    ### Loads Data
    ignore_missing_moltype = True if model in ['cox_omic', 'cox_moltype', 'cox_grade+moltype', 'all'] else False
    ignore_missing_histype = True if model in ['cox_histype', 'cox_grade', 'cox_grade+moltype', 'all'] else False
    all_dataset = getCleanAllDataset(dataroot=dataroot, ignore_missing_moltype=ignore_missing_moltype, 
                                     ignore_missing_histype=ignore_missing_histype)[1]
    model_feats = {'cox_omic':['TCGA ID', 'Histology', 'Grade', 'Molecular subtype', 'Histomolecular subtype'],
                   'cox_moltype':['Survival months', 'censored', 'codeletion', 'idh mutation'],
                   'cox_histype':['Survival months', 'censored', 'Histology'],
                   'cox_grade':['Survival months', 'censored', 'Grade'],
                   'cox_grade+moltype':['Survival months', 'censored', 'codeletion', 'idh mutation', 'Grade'],
                   'cox_all':['TCGA ID', 'Histomolecular subtype']}
    cv_results = []

    for k in pnas_splits.columns:
        pat_train = list(set(pnas_splits.index[pnas_splits[k] == 'Train']).intersection(all_dataset.index))
        pat_test = list(set(pnas_splits.index[pnas_splits[k] == 'Test']).intersection(all_dataset.index))
        feats = all_dataset.columns.drop(model_feats[model]) if model == 'cox_omic' or model == 'cox_all' else model_feats[model]
        train = all_dataset.loc[pat_train]
        test = all_dataset.loc[pat_test]

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(train[feats], duration_col='Survival months', event_col='censored', show_progress=False)
        cin = concordance_index(test['Survival months'], -cph.predict_partial_hazard(test[feats]), test['censored'])
        cv_results.append(cin)
        
        train.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(train))
        test.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(test))
        pickle.dump(train, open(os.path.join(ckpt_name, model, '%s_%s_pred_train.pkl' % (model, k)), 'wb'))
        pickle.dump(test, open(os.path.join(ckpt_name, model, '%s_%s_pred_test.pkl' % (model, k)), 'wb'))
        
    pickle.dump(cv_results, open(os.path.join(ckpt_name, model, '%s_results.pkl' % model), 'wb'))
    print("C-Indices across Splits", cv_results)
    print("Average C-Index: %f" % CI_pm(cv_results))


print('1. Omic Only. Ignore missing molecular subtypes')
trainCox(model='cox_omic', penalizer=1e-1)
print('2. molecular subtype only. Ignore missing molecular subtypes')
trainCox(model='cox_moltype', penalizer=0)
print('3. histology subtype only. Ignore missing histology subtypes')
trainCox(model='cox_histype', penalizer=0)
print('4. histologic grade only. Ignore missing histology subtypes')
trainCox(model='cox_grade', penalizer=0)
print('5. grade + molecular subtype. Ignore all NAs')
trainCox(model='cox_grade+moltype', penalizer=0)
print('6. All. Ignore all NAs')
trainCox(model='cox_all', penalizer=1e-1)

print('7. KM-Curves')
for model in ['cox_omic', 'cox_moltype', 'cox_histype', 'cox_grade', 'cox_grade+moltype', 'cox_all']:
    makeKaplanMeierPlot(ckpt_name='./checkpoints/surv_15_cox/', model=model, split='test')