import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import sklearn.neighbors


import cudf, cuml
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

print('cuML version', cuml.__version__)

train = cudf.read_csv('./data/digit-recognizer/train.csv')
print('train shape =', train.shape)

# CREATE 20% VALIDATION SET
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2, random_state=42)

# GRID SEARCH USING CROSS VALIDATION
for k in range(3, 6):
    time_start_1 = time.time()
    print('k =', k)
    oof = np.zeros(len(train))
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (idxT, idxV) in enumerate(skf.split(train.iloc[:, 1:], train.iloc[:, 0])):
        time_start_2 = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train.iloc[idxT, 1:], train.iloc[idxT, 0])
        # Better to use knn.predict() but cuML v0.11.0 has bug
        y_hat = knn.predict(train.iloc[idxV,1:])
        oof[idxV] = y_hat
        # y_hat_p = knn.predict_proba(train.iloc[idxV, 1:])
        # oof[idxV] = y_hat_p.to_pandas().values.argmax(axis=1)
        acc = (oof[idxV] == train.iloc[idxV, 0].to_array()).sum() / len(idxV)
        print(' fold =', i, ' acc =', acc, ' in ', time.time() - time_start_2)
    acc = (oof == train.iloc[:, 0].to_array()).sum() / len(train)
    print(' OOF with k =', k, 'ACC =', acc, ' in ', time.time() - time_start_1)
"""
train = pd.read_csv('./data/digit-recognizer/train.csv')
print('train shape =', train.shape)

# CREATE 20% VALIDATION SET
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2, random_state=42)

# GRID SEARCH USING CROSS VALIDATION
for k in range(3, 6):
    time_start_1 = time.time()
    print('k =', k)
    oof = np.zeros(len(train))
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (idxT, idxV) in enumerate(skf.split(train.iloc[:, 1:], train.iloc[:, 0])):
        time_start_2 = time.time()
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(train.iloc[idxT, 1:], train.iloc[idxT, 0])

        oof[idxV] = knn.predict(train.iloc[idxV, 1:])
        acc = (oof[idxV] == np.array(train.iloc[idxV, 0])).sum() / len(idxV)
        print(' fold =', i, ' acc =', acc, ' in ', time.time() - time_start_2)
    acc = (oof == train.iloc[:, 0].to_array()).sum() / len(train)
    print(' OOF with k =', k, 'ACC =', acc, ' in ', time.time() - time_start_1)
"""