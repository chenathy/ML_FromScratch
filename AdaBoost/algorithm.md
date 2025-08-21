<h1>AdaBoost</h1> 
<b>(short for Adaptive Boosting)</b> <br>
which is a boosting ensemble machine learning algorithm.

<h3>💡Core Idea</h3>
AdaBoost combines multiple weak learners (usually decision stumps — decision trees with a single split) into a <b>strong classifier</b>. It does this by:

1. Training a sequence of weak learners.
2. Each subsequent learner focuses more on the examples that were misclassified by the previous learners.
3. The final prediction is a <b>weighted vote</b> of all the weak learners.

<h3>🔢 How AdaBoost Works (Conceptually)</h3>
Given training data 
(𝑋, 𝑦):
1. Initialize weights 𝑤𝑖 = 1/𝑛 for all samples.
2. For each round 𝑡 = 1 … 𝑇:
   <ul>
        <li>Train a weak learner ℎ𝑡(𝑥) using the weighted dataset.</li>
        <li>Compute the weighted error: <br/>
            𝜖t = ∑ 𝑤𝑖 ⋅ ℓ(ℎ𝑡(𝑥) =/= 𝑦i)
        </li>
        <li>Compute the learner’s weight: <br/>
            𝛼𝑡 = 1/2 ⋅ ln( (1-𝜖t) / 𝜖t)
        </li>
        <li>Update the sample weights: <br/>
            𝑤𝑖 ← 𝑤𝑖 ⋅ 𝑒 ^ −𝛼𝑡⋅𝑦𝑖⋅ℎ𝑡(𝑥𝑖)
        </li>
        <li>Normalize 𝑤𝑖 to sum to 1.</li>
   </ul>
3. Final prediction is: <br/>
       𝐻(𝑥) = sign(∑ 𝛼𝑡 ℎ𝑡(𝑥))


Compute the learner’s weight:

Each subsequent learner focuses more on the examples that were misclassified by the previous learners.

The final prediction is a weighted vote of all the weak learners.


<h3>🪵  Decision Stump</h3>
A decision tree with just one node is called a decision stump. <br/>
It's nearly the simplest classifier we could imagine: the entire decision is based on a single binary feature of the example.
A <b>decision stump</b> is a very <b>simple decision tree</b> that:
<ul>
    <li>Has <b>only one split</b></li>
    <li>Uses <b>only one feature</b></li>
    <li>Divides data into <b>two parts</b></li>
</ul>

It’s literally a tree with:
<ul>
    <li>One root node</li>
    <li>Two leaf nodes</li>
</ul>

<b>📌 Why is it called a “stump”? </b><br/>
Because it’s a "cut-off" decision tree — just one level deep, like a stump of a tree.


<b>🔁 Polarity</b><br/>
So polarity tells the model: <br/>
Should "less than" mean positive? <br/>
Or should "greater than" mean positive?

Polarity in Decision Stump controls the direction of the inequality in the rule:

Polarity is needed because depending on the feature and threshold, one direction might be more correct than the other.

Polarity = 1 <br/>
```
If 𝑥𝑗 < threshold ⇒ predict +1 
OR 
If 𝑥𝑗 > threshold ⇒ predict -1 
```

Polarity = -1 <br/>
```commandline
If 𝑥𝑗 > threshold ⇒ predict +1 
OR 
If 𝑥𝑗 < threshold ⇒ predict -1 

```

<b>🧩 Summary </b>

| Polarity | Rule Applied         | Meaning |
|----------|----------------------|----------|
| +1       | if x < threshold → class -1 | Use < as the split  |
| -1       | if x > threshold → class -1 | Use > as the split  |






<h3> 🧠 Steps </h3> 
<b>Step 1: </b> <br>
When the algorithm is given data, it starts by Assigning equal weights to all training examples in the dataset.<br/>
These weights represent the importance of each sample during the training process.

<b>Step 2: </b> <br>
Here, this algorithm iterates with a few algorithms for a specified number of iterations (or until a stopping criterion is met). <br/>
The algorithm trains a weak classifier on the training data. <br/>
The weak classifier can be considered a model that performs slightly better than random guessing, such as a decision stump (a one-level decision tree).

<b>Step 3: </b> <br>
During each iteration, the algorithm trains the weak classifier on given training data with the current sample weights. <br/>
The weak classifier aims to minimize the classification error, weighted by the sample weights.

<b>Step 4: </b> <br>
After training the weak classifier, the algorithm calculates classifier weight based on the errors of the weak classifier. <br/>
A weak classifier with a lower error receives a higher weight. <br/>
Once the calculation of weight completes, the algorithm updates sample weights, and the algorithm gives assigns higher weights to misclassified examples so that more importance in subsequent iterations can be given.

<b>Step 5: </b> <br>
After updating the sample weights, they are normalized so that they sum up to 1 and Combine the predictions of all weak classifiers using a weighted majority vote. <br/>
The weights of the weak classifiers are considered when making the final prediction.

<b>Step 6: </b> <br>
Finally, Steps 2–5 are repeated for the specified number of iterations (or until the stopping criterion is met), with the sample weights updated at each iteration. <br/> 
The final prediction is obtained by aggregating the predictions of all weak classifiers based on their weights.