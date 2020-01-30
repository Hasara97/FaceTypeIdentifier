import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')

print(data.shape)
print(target.shape)

from sklearn.neighbors import KNeighborsClassifier

algorithm=KNeighborsClassifier()

algorithm.fit(data,target)

import joblib

joblib.dump(algorithm,'KNN_model.sav') 
