
**Step 1**  
Initialize the weights to zero or small random values.

---
**Step 2**  
For each iteration: 
- Calculate the dot product of the input features and the weights.
- Pass the dot product through the sigmoid function to get the predicted probability of the positive class.
- Calculate the error between the predicted probability and the true class label.
- Calculate the gradient of the weights using the error and the input features.
- Update the weights using the gradient and the learning rate.
  
---
**Step 3**  
Repeat step 2 for a specified number of iterations or until convergence is achieved.

---
**Step 4**
To make predictions on new data:  
- Calculate the dot product of the input features and the weights.
- Pass the dot product through the sigmoid function to get the predicted probability of the positive class.
- Use a threshold (usually 0.5) to convert the predicted probability to a binary class label (0 or 1).

