Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for classification, regression, and even outlier detection tasks.  
It's particularly effective in high-dimensional spaces and is well-suited for situations where the data is not linearly separable.
 

The basic idea behind SVM is to find a hyperplane that best separates the data into different classes.  
The "best" hyperplane is the one that maximizes the margin, which is the distance between the hyperplane and the nearest data point from each class. SVM can be used with both linear and non-linear data by employing different types of kernels.

##key concepts

--------

* ###Support Vectors
data points that are closest to the hyperplane and influence its position and orientation.

-------

* ###Hyperplane
decision boundary that separates the data into different classes.

In a binary classification problem, the hyperplane is the one with the maximum margin.

-------

* ###Margin  
The margin is the distance between the decision boundary (i.e., the hyperplane in SVM) and the nearest data points from both classes (i.e., the support vectors). 

The goal of SVM is to find the decision boundary that maximizes the margin between the support vectors.
A larger margin implies better generalization performance of the SVM model.  

----

* ###Kernel function:  
The kernel function is used to transform the input data into a higher-dimensional space, making it possible to find a hyperplane that can separate non-linearly separable data.  

The kernel function can be linear, polynomial, or radial basis function (RBF) kernel, among others.  

The choice of kernel function depends on the characteristics of the data.

----

* ###Hyperparameters:  

Hyperparameters are parameters that are set before the learning process begins and can greatly affect the performance of the SVM model.  

In addition to the cost parameter C and the kernel function, other hyperparameters in SVM include gamma, degree, and coefficient0, among others.  

The hyperparameters can be tuned using cross-validation techniques to find the optimal values.

---

* ###Multiclass classification:  
SVM is originally designed for binary classification problems.  

However, it can be extended to multiclass classification using various methods such as one-vs-one and one-vs-all approaches.


--------------------

### SVM Error = Margin Error + Classification Error
* Margin Error: ​ ∣∣w∣∣
* Classification Error: max(0,1−y i ​(w⋅x i ​ −b)))

The objective function for a linear Support Vector Machine (SVM) with regularization is typically formulated as follows:  
minimize( 1/2 *​ ∣∣w∣∣ ^ 2 + C⋅∑ ​ max(0,1−y i ​(w⋅x i ​ −b)))

C: regularization parameter