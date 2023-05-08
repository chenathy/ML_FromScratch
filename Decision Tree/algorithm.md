<h4>Decision Tree Algorithm Steps </h4>


<ol>
    <li><b>Start with the entire dataset as the root node.</b></li>
<br>
    <li><b>Choose the best attribute to split the data into subsets.</b></li>
    This is done by calculating the <u>Information Gain</u> or <u>Gini Impurity</u> for each attribute 
    and choosing the attribute with the highest score.
<br>
<br>
    <li><b>Create a new branch for each possible value of the selected attribute, and partition the data accordingly.</b></li>
<br>
    <li><b>Repeat steps 2 and 3 recursively for each branch, until all data points in a given branch belong to the same class or all attributes have been used.
</b></li>
<br>
<br>
    <li><b>Assign the majority class in each leaf node.</b></li>
<br>
<br>
    <li><b>Prune the tree by removing branches that do not improve the overall accuracy of the tree on a validation set.</b></li>
<br>
    <li><b>
Optionally, repeat steps 5 and 6 until the accuracy of the tree stops improving.</b></li>


</ol> 

