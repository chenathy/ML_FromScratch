
1. Start with the entire dataset as the root node.  
---

2. Choose the best attribute to split the data into subsets. 
This is done by calculating the information gain or Gini impurity for each attribute
and choosing the attribute with the highest score.

---

3. Create a new branch for each possible value of the selected attribute, and partition the data accordingly.

---

4. Repeat steps 2 and 3 recursively for each branch, until all data points in a given branch belong to the same class or all attributes have been used.

---

5. Assign the majority class in each leaf node.

---

6. Prune the tree by removing branches that do not improve the overall accuracy of the tree on a validation set.

---

7. Optionally, repeat steps 5 and 6 until the accuracy of the tree stops improving.

<br>
<br>

<h4>What is Gini Impurity?<h4/>
Gini impurity is a measure of the probability of incorrectly classifying a randomly chosen element in a dataset  
If it were randomly labeled according to the distribution of labels in the dataset.
<br>
<br>
For a given dataset, the Gini impurity is calculated as follows:

Compute the probability p_i of each class i in the dataset.

Compute the Gini impurity Gini(S) as follows:
```
Gini(S) = 1 - Σ p_{i}^2
```


where the summation is taken over all classes i.

<br>
<br>
The Gini impurity is a number between 0 and 1, where a value of 0 indicates that the dataset is perfectly classified  
(i.e., all elements belong to the same class) and a value of 1 indicates that the dataset is evenly distributed across all classes.

<br>
<br>
When building a decision tree, the goal is to minimize the Gini impurity of the subsets created by each split in the tree.  
A split that separates the data into pure subsets

(i.e., subsets where all elements belong to the same class) will have a Gini impurity of 0, and is therefore preferred over a split that creates impure subsets.

<br>
<h4>What is Information Gain</h4>

Information gain is a measure of the reduction in entropy (or Gini impurity) achieved by partitioning a dataset based on a given attribute.

The basic idea behind information gain is that by splitting a dataset into subsets based on an attribute, we should obtain subsets that are more homogeneous with respect to the target variable than the original dataset.   
The information gain of a split is therefore a measure of how much more we know about the target variable (i.e., how much entropy we have reduced) by knowing the value of the attribute used to split the data.  

Given a dataset S and a split on attribute A that creates two subsets S1 and S2, the information gain of the split is defined as follows:

Information Gain(A) = Entropy(S) - [Weight(S1) x Entropy(S1) + Weight(S2) x Entropy(S2)]

where Entropy(S) is the entropy of the original dataset S, Weight(S1) and Weight(S2) are the weights of the subsets S1 and S2 relative to the original dataset S, and Entropy(S1) and Entropy(S2) are the entropies of the subsets S1 and S2.

In the case of Gini impurity, we can replace "entropy" with "Gini impurity" in the above formula to calculate the information gain for a split based on Gini impurity.

The attribute with the highest information gain is typically chosen as the splitting attribute, as it is expected to provide the most useful information for classification.






