PCA is a dimensionality reduction technique commonly used in machine learning and data analysis.  
Its primary goal is to reduce the dimensionality of a dataset while preserving as much of the original variance as possible.  
This is achieved by transforming the data into a new coordinate system, where the new axes (principal components) are orthogonal to each other and are ranked by the amount of variance they explain.


PCA has various applications, such as reducing noise in data, visualizing high-dimensional data, and improving the performance of machine learning algorithms by reducing multicollinearity among features.  
It's a valuable tool for exploratory data analysis and feature engineering.

Here's a high-level overview of how PCA works:

1. **Standardize the Data**:  
Before applying PCA, it's essential to standardize or normalize the data to have a mean of 0 and a standard deviation of 1.  
This step is crucial because PCA is sensitive to the scale of the variables.

<br>

2. **Calculate the Covariance Matrix**:   
PCA involves finding the covariance matrix of the standardized data.   
The covariance matrix represents the relationships between pairs of variables in the dataset.
X · XT

<br>

3. **Compute the Eigenvectors and Eigenvalues**:  
The next step is to calculate the eigenvectors and eigenvalues of the covariance matrix.  
Eigenvectors are the directions along which the data varies the most (the principal components), and eigenvalues represent the amount of variance explained by each principal component.

<br>

4. **Sort and Select Principal Components**:  
Sort the eigenvalues in descending order.  
The corresponding eigenvectors represent the principal components.  
You can choose to keep a subset of these components based on the desired level of variance retention.  
__reduce dimensionality__  
__form feature vector__  
__first r of m-dimensions__

<br>

5. **Transform the Data**:   
Create a new dataset by projecting the original data onto the selected principal components.  
This transformation reduces the dimensionality of the data.  
__Y = P · XT

<br>

<h3>Orthonormal</h3>
||v|| = ||u|| = 1  
vT · u = ∑ vi · ui = 0 ==> v ⊥ u

<h3>Orthogonal Matrix</h3>
U: Orthogonal square matrix  
UT · U = U-1 · U  =  I  
U-1 = UT


<h3> Symmetric Matrix </h3>
A: Symmetric matrix  <br>
D: Diagonal matrix  <br>
U: Orthogonal matrix  <br>
A = U·D·UT
