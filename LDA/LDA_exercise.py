# Load Modules
import pandas as pd
from LDA.lda import LDA

k = 1  # Number of dimensions for the reduced feature space

# Data Prep
#### Read Data
data_loc = 'IRIS.csv'
iris = pd.read_csv(data_loc)
iris = iris.sample(frac=1, random_state=1234)
X = iris.drop('species', axis=1)
y = iris['species']

# Linear Discrimination Analysis Projection
lda = LDA()
X_projected = lda.fit_transform(X, y)
print(f'LDA projected data shape: {X_projected.shape}')
