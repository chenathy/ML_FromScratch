###Purpose
It's a binary classifier that decides whether an input belongs to one class or another.

###Structure:

**Input layer**: 
𝑥1 , 𝑥2, ... , 𝑥𝑛

**Weights**: 
𝑤1, 𝑤2, ... , 𝑤𝑛

**Bias**: 
𝑏

**Activation function** (usually a step function):

            1  if  𝑤 ⋅ 𝑥 + 𝑏 > 0  
output = {

            0  otherwise


###Learning Rule
(Perceptron update rule):
Adjust weights as: 

𝑤𝑖 : = 𝑤𝑖 + Δ𝑤𝑖 
where  Δ𝑤𝑖 = 𝜂 (𝑦 − 𝑦^) 𝑥𝑖

 
𝜂 is the learning rate, 

𝑦 is the true label, and 

𝑦^ is the predicted label.


###Limitations:
Can only solve linearly separable problems.

Fails for cases like the XOR problem.