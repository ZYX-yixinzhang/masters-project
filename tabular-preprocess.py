"""
    This file includes the reading and preprocessing of tabular data on ADNI dataset,
    it includes:
        read data and select useful features
        visualise useful terms
        data impute
        under / over sampling
        train-test split
        store data in csv files
"""
import pandas as pd
from sklearn.impute import KNNImputer

files = pd.read_csv('../dataset/dataset.csv')
dataset = files[['PTID', 'DX_bl', 'APOE4', 'VISCODE', 'ADAS13', 'FDG', 'TAU', 'PTAU', 'CDRSB', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting'
               , 'RAVLT_perc_forgetting', 'FAQ', 'MOCA', 'Hippocampus', 'AGE', 'PTGENDER', 'PTEDUCAT']]
for n in dataset.columns:
    print(dataset[n].value_counts())
threshold = int(len(dataset.columns)*0.6)
dataset.dropna(thresh = threshold, inplace=True)
print('missing values for registered ids', dataset['PTID'].isnull().sum(), '\nmissing values for labels', dataset['DX_bl'].isnull().sum())
dataset.dropna(subset=['DX_bl', 'VISCODE', 'APOE4'], inplace=True)
dataset = dataset[(dataset['DX_bl']=='AD') | (dataset['DX_bl']=='EMCI') | (dataset['DX_bl']=='LMCI')]
print(dataset)

dataset = dataset.replace('<8', 8).replace('<80', 80).replace('>1300', 1300).replace('>120', 120)
dataset = dataset.replace('EMCI', 'MCI').replace('LMCI', 'MCI')
val = ['APOE4', 'ADAS13', 'FDG', 'TAU', 'PTAU', 'CDRSB', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'FAQ', 'MOCA', 'Hippocampus', 'AGE', 'PTEDUCAT']
dataset_ad = dataset[dataset['DX_bl']=='AD']
dataset_ad_num = dataset_ad[val]#.values()
imputer = KNNImputer(n_neighbors=5)
dataset_ad_num = imputer.fit_transform(dataset_ad_num.to_numpy())
for n in range(len(val)):
    dataset_ad[val[n]] = dataset_ad_num[:, n]
dataset_mci = dataset[dataset['DX_bl']=='MCI']
dataset_mci_num = dataset_mci[val]#.values()
dataset_mci_num = imputer.fit_transform(dataset_mci_num.to_numpy())
for n in range(len(val)):
    dataset_mci[val[n]] = dataset_mci_num[:, n]
dataset = pd.concat([dataset_ad, dataset_mci])

# preprocessing of GRU (sequencial data)
# ids = dataset['PTID'].unique()
# dct = {'PTID':ids, 'bl': None, 'm06': None, 'm12': None, 'm24': None}
# labels = pd.DataFrame(dct)
# labels.set_index('PTID', inplace=True)
# stemps = {'bl':0, 'm06':0, 'm12':0, 'm24':0}
# map_ = {'AD':1., 'MCI':0.}
# for n in range(len(dataset)):
#     if dataset.iloc[n]['VISCODE'] in stemps:
#         labels.loc[dataset.iloc[n]['PTID']][dataset.iloc[n]['VISCODE']]=map_[dataset.iloc[n]['DX_bl']]
#     #break
# labels.dropna(inplace=True)
# labels = labels.astype(float)
# labels = labels[:308]#.index[19:] == dataset_ft.index19:
# labels

import numpy as np
dataset_lb = dataset[(dataset['VISCODE']=='m12') | (dataset['VISCODE']=='m24')]
dataset_ft = dataset[dataset['VISCODE'] == 'bl']
# dataset_ft
dataset_ft.insert(dataset_ft.shape[1], 'label', np.nan)
dataset_ft.set_index('PTID', inplace=True)

for n in dataset_lb.to_numpy():
    dataset_ft.loc[n[0],'label']=n[1]
dataset_ft.dropna(subset=['label'], inplace=True)
dataset_ft.drop(['DX_bl', 'VISCODE'], axis=1, inplace=True)

# preprocessing of GRU (sequencial data)
# dataset_ft = dataset[dataset['VISCODE'] == 'bl']
# dataset_ft
# dataset_ft.insert(dataset_ft.shape[1], 'label', np.nan)
# dataset_ft.set_index('PTID', inplace=True)
# dataset_ft.insert(dataset_ft.shape[1], 'm06', np.nan)
# dataset_ft_1 = dataset[dataset['VISCODE'] == 'm06']
# dataset_ft_1
# dataset_ft_1.insert(dataset_ft_1.shape[1], 'label', np.nan)
# for n in labels.index:
#     dataset_ft.loc[n,'label']=True
# for n in dataset_ft_1.to_numpy():
#     dataset_ft.loc[n[0],'m06']=True
# dataset_ft.dropna(subset=['label', 'm06'], inplace=True)
# dataset_ft.drop(['DX_bl', 'VISCODE'], axis=1, inplace=True)
# dataset_ft

dataset_ft = dataset_ft.replace('Male', 1.).replace('Female', 0.)
dataset_ft = dataset_ft.replace('AD', 1.).replace('MCI', 0.)
df_mci = dataset_ft[dataset_ft.label == 0.]
df_ad = dataset_ft[dataset_ft.label == 1.]
df_mci_sampled = df_mci.sample(800, replace=True)
df_ad_sampled = df_ad.sample(800, replace=True)

# preprocessing of GRU (sequencial data)
# demographic = dataset_ft[['APOE4', 'PTGENDER', 'PTEDUCAT']]
# demographic
# dataset_ft.drop(['m06', 'APOE4', 'PTGENDER', 'PTEDUCAT', 'label'], axis=1, inplace=True)
# dataset_ft_1.drop(['DX_bl', 'PTGENDER', 'APOE4', 'PTEDUCAT', 'VISCODE', 'label'], axis=1, inplace=True) #
# dataset_ft = dataset_ft.replace('Male', 1.0).replace('Female', 0.0)
# dataset_ft_1 = dataset_ft_1.replace('Male', 1.0).replace('Female', 0.0)
# dataset_ft
# feature_0 = dataset_ft.T.to_dict()
# dataset_ft_1.set_index('PTID', inplace=True)
# feature_1 = dataset_ft_1.T.to_dict()
# features = {}
# for key, item in feature_0.items():
#     feature_item = []
#     feature_item_06 = []
#     for n in dataset_ft.columns:
#         feature_item.append(item[n])
#     for n in dataset_ft_1.columns:
#         feature_item_06.append(feature_1[key][n])
#     features[key] = np.array([feature_item, feature_item_06], dtype=np.float32)
# features

dataset_ft = pd.concat([df_mci_sampled, df_ad_sampled])
dataset_ft = dataset_ft.sample(frac=1)

train_test_split = 0.1
train = dataset_ft[:int(len(dataset_ft)*(1-0.1))]
test = dataset_ft[int(len(dataset_ft)*(1-0.1)):]

print(train)
print(test)

print(dataset_ft.describe())
import seaborn as sns

sns.pairplot(dataset_ft)
# train.to_csv('./train.csv')
# test.to_csv('./test.csv')