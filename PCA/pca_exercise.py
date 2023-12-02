# Load Modules
import pandas as pd
from PCA.pca import PCA



# Data Prep
#### Read Data
data_loc = 'IRIS.csv'
iris = pd.read_csv(data_loc)
iris = iris.sample(frac=1, random_state=1234)
X = iris.drop('species', axis=1)
y = iris['species']

# PCA Transform Data
pca = PCA(X, n_components=3)
pcaTransformed_df = pca.transform_data()

print(f'PCA transformed data columns: {pcaTransformed_df.shape[1]}')
print('Preview of PCA data:')
print(pcaTransformed_df.head())