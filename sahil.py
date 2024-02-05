import pandas as pd
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.decomposition import PCA


wine = load_wine()
data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['target'] = wine.target



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



print("\n \nThe list of all columns : \n")
print(data.columns)



print("\n \nThe first 5 rows of the data : \n")
print(data.head())



print("\n \nStep 1: Handling Missing Values \n")
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
print(data_imputed.head())



print("\n \nStep 2: Normalized Data \n")
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)
print(data_normalized.head())



print("\n \nStep 3: De-Normalized Data \n")
data_denormalized = pd.DataFrame(scaler.inverse_transform(data_normalized), columns=data_normalized.columns)
print(data_denormalized.head())



print("\n \nStep 4: Percentile Feature Selection Results \n")
selector_percentile = SelectPercentile(percentile=30)
data_percentile = selector_percentile.fit_transform(data_denormalized, data_denormalized['alcohol'])
print(data_percentile[:5])



print("\n \nStep 5: K Best Feature Selection Results \n")
selector_kbest = SelectKBest(k=3)
data_kbest = selector_kbest.fit_transform(data_denormalized, data_denormalized['alcohol'])
print(data_kbest[:5])



print("\n \nStep 6: Variance threshold Feature Selection Results \n")
selector_variance = VarianceThreshold(threshold=0.1)
data_variance = selector_variance.fit_transform(data_denormalized)
print(data_variance[:5])



print("\n \nStep 7: PCA Feature Selection Results \n")
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_denormalized)
print(data_pca[:5])
