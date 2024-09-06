import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# read the data
train = pd.read_csv('./train.csv')
train = train.sample(frac=1)
train = train.set_index('PTID')
test = pd.read_csv('./test.csv')
test = test.sample(frac=1)
test = test.set_index('PTID')
label_tr = train['label'].replace('EMCI', 0.).replace('LMCI', 0.).replace('AD', 1.).replace('MCI', 0.)#.values
label_ts = test['label'].replace('EMCI', 0.).replace('LMCI', 0.).replace('AD', 1.).replace('MCI', 0.)#.values
train = train.drop('label', axis=1)
test = test.drop('label', axis=1)
clf = RandomForestClassifier(n_estimators = 20) 
clf.fit(train, label_tr)
y_pred = clf.predict(test)
print("accuracy of RF: ", metrics.accuracy_score(label_ts, y_pred))