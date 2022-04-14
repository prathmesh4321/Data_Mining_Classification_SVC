import pandas as pd
import pickle
import pickle_compat
pickle_compat.patch()
from sklearn import decomposition
from sklearn import preprocessing
from train import glucoseFeatures

with open("model.pkl", 'rb') as file:
        model = pickle.load(file) 
        test_data = pd.read_csv('test.csv', header=None)
    
feature_cgm=glucoseFeatures(test_data)
std_scaler_fit = preprocessing.StandardScaler().fit_transform(feature_cgm)
    
pca = decomposition.PCA(n_components=5)
pca_fit = pca.fit_transform(std_scaler_fit)
    
results = model.predict(pca_fit)
pd.DataFrame(results).to_csv("Results.csv", header=None, index=False)
