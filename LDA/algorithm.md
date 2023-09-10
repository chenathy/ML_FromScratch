<h2>Linear Discriminant Analysis</h2>

<b>LDA</b> is a dimensionality reduction technique commonly used for feature extraction and classification tasks.   
It is also known as Fisher's Linear Discriminant Analysis (FLDA) or simply Fisher's LDA.

<br>

LDA aims to find a linear combination of features that maximizes the separation between classes while minimizing the variation within each class.   
It assumes that the data follows a Gaussian distribution and that the classes have identical covariance matrices.

<br>

 LDA assumes linearity and Gaussian distribution, so it may not be suitable for data with complex nonlinear relationships or non-Gaussian distributions.  
 In such cases, nonlinear dimensionality reduction techniques like Kernel Discriminant Analysis (KDA) or nonlinear manifold learning algorithms like t-SNE may be more appropriate.

<h2>Core</h2>
- <h4>2 classes</h4>  
Sb (between class): (µ1 − µ2)^2   
Sw (within class): ( s1^2 + s2^2 )  
max (µ1 − µ2)^2 / ( s1^2 + s2^2 )  


- <h4>general</h4> (more than 3 classes)  
Sb = ∑ Ni · (µi - µ) · (µi - µ)'  
Sw = ∑ ∑ (X - µi) · (X - µi)' = ∑ cov(Xi) * Ni   
max Sb · Sw-1


<br>

<h5>The main steps of LDA are as follows:</h5>
<ol>
    <li>Compute the mean vectors of each class.</li>
    <li>Compute the scatter matrices:
    <ul>
        <li>Within-class scatter matrix (Sw): Measures the variation within each class.</li>
        <li>Between-class scatter matrix (Sb): Measures the separation between classes.</li>
    </ul>
    </li>
    <li>Compute the eigenvalues and eigenvectors of the matrix Sw^(-1) * Sb.</li>
    <li>Sort the eigenvalues in descending order and select the corresponding eigenvectors.</li>
    <li>Project the data onto the subspace spanned by the selected eigenvectors.</li>
</ol>

<br>

<b>EigenValue</b> & <b>EigenVector</b>  
 A: square matrix (n x n)  
 λ: scalar value  
 X: vector (n x 1)  
<br>
 A · X = λ · X

<br>
<br>

<b>Important issue: </b>  
For high dimensional data (i.e., d is large),   
the centered data often do not fully span all d dimensions,  
thus making rank(Sw) = rank(X) < d  (which implies that Sw is singular).