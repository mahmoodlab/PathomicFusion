import os

import pandas as pd
import numpy as np

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


def getCleanGBMLGG(dataroot='./data/TCGA_GBMLGG/', ignore_missing_moltype=False, ignore_missing_histype=False, use_rnaseq=False, use_ag=False):
    ### 1. Joining all_datasets.csv with grade data. Looks at columns with misisng samples
    metadata = ['Histology', 'Grade', 'Molecular subtype', 'TCGA ID', 'censored', 'Survival months']
    all_dataset = pd.read_csv(os.path.join(dataroot, 'all_dataset.csv')).drop('indexes', axis=1)
    all_dataset.index = all_dataset['TCGA ID']

    all_grade = pd.read_csv(os.path.join(dataroot, 'grade_data.csv'))
    all_grade['Histology'] = all_grade['Histology'].str.replace('astrocytoma (glioblastoma)', 'glioblastoma', regex=False)
    all_grade.index = all_grade['TCGA ID']
    all_grade = all_grade.rename(columns={'Age at diagnosis': 'Age'})
    all_grade['Gender'] = all_grade['Gender'].replace({'male':0, 'female': 1})
    assert pd.Series(all_dataset.index).equals(pd.Series(sorted(all_grade.index)))

    all_dataset = all_dataset.join(all_grade[['Histology', 'Grade', 'Molecular subtype', 'Age', 'Gender']], how='inner')
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

    if use_ag == 0:
        metadata.extend(['Age', 'Gender'])

    all_dataset['censored'] = 1 - all_dataset['censored']
    return metadata, all_dataset

def getCleanKIRC(dataroot='./', rnaseq_cutoff='all', cnv_cutoff=7.0, mut_cutoff=5.0):
    ### Clinical variables
    clinical = pd.read_table(os.path.join(dataroot, './kirc_tcga_pan_can_atlas_2018_clinical_data.tsv'), index_col=2)
    clinical.index.name = None
    clinical['censored'] = clinical['Overall Survival Status']
    clinical['censored'] = clinical['censored'].replace('LIVING', 1)
    clinical['censored'] = clinical['censored'].replace('DECEASED', 0)
    clinical['censored'] = 1-clinical['censored']

    ### Select RNAseq Features
    rnaseq = pd.read_table(os.path.join(dataroot, 'data_RNA_Seq_v2_mRNA_median_Zscores.txt'), index_col=0)
    rnaseq = rnaseq[rnaseq.index.notnull()]
    rnaseq = rnaseq.drop(['Entrez_Gene_Id'], axis=1)
    rnaseq.index.name = None
    rnaseqDEGs = pd.read_csv(os.path.join(dataroot, 'dataDEGs_kirc.csv'), index_col=0)
    rnaseqDEGs = rnaseqDEGs.sort_values(['PValue', 'logFC'], ascending=False)
    rnaseq_cutoff = rnaseqDEGs.shape[0] if isinstance(rnaseq_cutoff, str) else rnaseq_cutoff
    rnaseq = rnaseq.loc[rnaseq.index.intersection(rnaseqDEGs.index)].T
    rnaseq.columns = [g+"_rnaseq" for g in rnaseq.columns]

    ### Select CNV Features
    cnv = pd.read_table(os.path.join(dataroot, 'data_CNA.txt'), index_col=0)
    cnv = cnv[cnv.index.notnull()]
    cnv = cnv.drop(['Entrez_Gene_Id'], axis=1)
    cnv.index.name = None
    cnv_freq = pd.read_table(os.path.join(dataroot, 'CNA_Genes.txt'), index_col=0)
    cnv_freq = cnv_freq[['CNA', 'Profiled Samples', 'Freq']]
    cnv_freq['Freq'] = cnv_freq['Freq'].str.rstrip('%').astype(float)
    cnv_cutoff = cnv_freq.shape[0] if isinstance(cnv_cutoff, str) else cnv_cutoff
    cnv_freq = cnv_freq[cnv_freq['Freq'] >= cnv_cutoff]
    cnv = cnv.loc[cnv.index.intersection(cnv_freq.index)].T
    cnv.columns = [g+"_cnv" for g in cnv.columns]
                                 
    mut = clinical[['Patient ID']].copy()
    for tsv in os.listdir(os.path.join(dataroot, 'muts')):
        if tsv.endswith('.tsv'):
            mut_samples = pd.read_table(os.path.join(dataroot, 'muts', tsv))['Patient ID']
            mut_gene = tsv.split('_')[2].rstrip('.tsv')+'_mut'
            mut[mut_gene] = 0
            mut.loc[mut.index[:-3].isin(mut_samples), mut_gene] = 1
    mut = mut.drop(['Patient ID'], axis=1)
    
    omic_features = rnaseq.join(cnv, how='inner').join(mut, how='inner')
    return omic_features

