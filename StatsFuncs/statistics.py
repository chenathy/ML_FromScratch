import numpy as np
import pandas as pd

class Statistics:

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


    @staticmethod
    def logit(p):
        return np.exp(p / (1 - p))


    @staticmethod
    def one_hot_encoder(single_column_df):

        # list unique values
        unique_vals = list(np.unique(single_column_df))

        if len(unique_vals) > 2:

            # output columns names
            col_names = [single_column_df.name + '_' + str(i) for i in unique_vals]

            # generate one hot encoded dataframe
            col_arrays = np.array([
                [1 if str(i) == j.split('_')[1] else 0 for i in single_column_df] for j in col_names
            ]).T
            col_df = pd.DataFrame(col_arrays, columns=col_names)

            # Drop the very 1st column
            col_df.drop(col_names[0], axis=1, inplace=True)

            return col_df
        else:
            return single_column_df

    @staticmethod
    def merge_encoded_df(df, column_name):

        encoded_df = Statistics.one_hot_encoder(df[column_name])

        df = df.drop(column_name, axis=1)

        df.reset_index(drop=True, inplace=True)

        df_cat = pd.concat([df, encoded_df], axis=1)

        return df_cat


    @staticmethod
    def assign_bias(features_df):

        column_names = list(features_df.columns)
        column_names.insert(0, 'bias')

        features_df = np.hstack((np.ones((features_df.shape[0], 1)), features_df))
        features_df = pd.DataFrame(features_df, columns=column_names)

        return features_df


    @staticmethod
    def scale_features(train_df, test_df):

        for col in train_df.columns:
            print(f'normalizing feature {col} ...')
            mean_train = np.mean(train_df[col])
            std_train = np.std(train_df[col])

            # standardize training data
            train_df[col] = (train_df[col] - mean_train) / std_train

            # standardize testing data
            test_df[col] = (test_df[col] - mean_train) / std_train

        print('Finished Normalizing both Train and Test data')
        return train_df, test_df



    @staticmethod
    def confusion_matrix(y_true, y_pred):

        # convert them into numpy array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Count the numbers for
        ## True Positive
        ## False Negative
        ## False Postive
        ## True Negative


                                   # | # ACTUAL # | # ACTUAL # |
                                   # |  Postive   |  Negative  |
        #### PREDICTED   Positive    |    TP      |     FP     |
        #### PREDICTED   Negative    |    FN      |     TN     |


        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        # Create a Pandas dataframe to store the results
        cm = pd.DataFrame({
            'Actual 1': [tp, fp],
            'Actual 0': [fn, tn]
        }, index=['Predicted 1', 'Predicted 0'])

        return cm


    @staticmethod
    def accuracy_metrics(y_true, y_pred):

        # Create Confusion Matrix
        cm = Statistics.confusion_matrix(y_true, y_pred)

        # Calculate true positives, false positives, and false negatives
        tp = cm.loc['Predicted 1', 'Actual 1']
        fp = cm.loc['Predicted 1', 'Actual 0']
        fn = cm.loc['Predicted 0', 'Actual 1']

        # Calculate accuracy, recall, and F1 score
        accuracy = np.mean(y_true == y_pred)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Create a Pandas dataframe to store the results
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score'],
            'Value': [accuracy, recall, precision, f1_score]
        })

        return metrics_df


    @staticmethod
    def gini_impurity(class_variable):
        _, counts = np.unique(class_variable, return_counts=True)
        probs = counts/len(class_variable)
        return 1 - np.sum(probs ** 2)


    @staticmethod
    def entropy(class_variable):
        _, counts = np.unique

