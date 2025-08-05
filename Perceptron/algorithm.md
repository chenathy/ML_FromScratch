<h3>Purpose</h3>
It's a binary classifier that decides whether an input belongs to one class or another.

<h3>Structure:</h3>

<b>Input layer</b>: 
𝑥1 , 𝑥2, ... , 𝑥𝑛

<b>Weights</b>: 
𝑤1, 𝑤2, ... , 𝑤𝑛

<b>Bias</b>: 
𝑏

<b>Activation function</b> (usually a step function):

            1  if  𝑤 ⋅ 𝑥 + 𝑏 > 0  
output = {

            0  otherwise


<h3>Learning Rule</h3> (Perceptron update rule):

Adjust weights as: 

𝑤𝑖 : = 𝑤𝑖 + Δ𝑤𝑖 
where  Δ𝑤𝑖 = 𝜂 (𝑦 − 𝑦^) 𝑥𝑖

 
𝜂 is the learning rate, 

𝑦 is the true label, and 

𝑦^ is the predicted label.


<h3>Limitations:</h3>
Can only solve linearly separable problems.

Fails for cases like the XOR problem.