from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the Boston Housing dataset
boston = load_boston()

# Split the dataset into training and test sets
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)

# Create a LinearRegression object
model_lr = LinearRegression()

# Fit the model on the training data
model_lr.fit(X_train, y_train)

# Print the coefficients and intercept of the model
print('Coefficients:', model_lr.coef_)
print('Intercept:', model_lr.intercept_)

# Evaluate the model on the test data
score = model_lr.score(X_test, y_test)
print('R-squared score:', score)



