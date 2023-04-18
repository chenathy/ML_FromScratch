
<h4>Custom Statistics Function including: </h4>

- <b>sigmoid</b> (z)  
sigmoid: 
$$ y = g(z) = \frac{1}{1 + e^z} = \frac{e^z}{1 + e^z} $$

- <b>logit</b> (p)

- <b>merge_encoded_df</b> (df, column_name)

- <b>assign_bias</b> (features_df)

- <b>scale_features</b> (train_df, test_df)

- <b>confusion_matrix</b> (y_true, y_pred)  


| Confusion Matrix |            |   Actual   |  Actual  |
|------------------|:----------:|:----------:|:--------:|
|                  |            |  Positive  | Negative |
| **Predicted**    |  Positive  |     TP     |    FP    |
| **Predicted**    |  Negative  |     FN     |    TN    |

- <b>accuracy_matrix</b> (y_true, y_pred)

- <b>gini_impurity</b> (class_variable) 

- <b>entropy</b> (class_variable)

