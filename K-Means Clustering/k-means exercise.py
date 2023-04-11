import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Load data
data_loc = 'IRIS.csv'
iris_df = pd.read_csv(data_loc)

X_df = iris_df.drop('species', axis=1)
y_df = iris_df['species']

# Train & Test Spliting
train_ratio = 0.8
train_rows = int(X_df.shape[0] * train_ratio)

X_train = X_df.iloc[:train_rows]
X_test = X_df.iloc[train_rows:]

y_train = y_df.iloc[:train_rows]
y_test = y_df.iloc[train_rows:]


# Step 1. Initialization:
# - The first step is to choose the number of clusters (K) and randomly initialize K centroids within the range of the dataset.
# - The centroids can be initialized using various methods such as random selection or K-means++ algorithm.

def centroids_initialize(k, dataframe):

    print(f'Dataframe has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns.')
    random_indexes =np.random.randint(
        low=dataframe.index.min(),
        high=dataframe.index.max(),
        size=k
    )

    centroids_df = dataframe.iloc[random_indexes]

    return centroids_df

K = 3

centroids_random_k = centroids_initialize(K, X_train)

# 2. Assigning data points to clusters:
# - In this step, each data point is assigned to the nearest centroid based on its distance.
# - The distance can be calculated using various distance metrics such as Euclidean distance, Manhattan distance, or cosine distance.

def distance_cal(dataframe, centroids_df, distance_type = 'Euclidean'):

    """
    :param dataframe: m x col
    :param centroids_df: k x col
    :return: distance matrix: m x k
    """

    # convert dataframe and cetroids_df into matrix (numpy)
    if not isinstance(dataframe, np.ndarray):
        dataframe = dataframe.to_numpy()

    if not isinstance(centroids_df, np.ndarray):
        centroids_df = centroids_df.to_numpy()


    # `K` & `rows` to iterate for distance calculation
    if dataframe.ndim == 1:
        rows = 1
    else:
        rows = dataframe.shape[0]


    if centroids_df.ndim == 1:
        K = 1
    else:
        K = centroids_df.shape[0]


    # distance
    if distance_type == 'Euclidean':

        distance = np.sqrt(
            [
                np.sum([
                    np.square(dataframe[row, :] - centroids_df[k, :])
                    for k in range(K)
                ],
                    axis=1
                )
                for row in range(rows)
            ]
        )

    elif distance_type == 'Manhattan':
        distance = np.sum([
            [np.abs(dataframe[row, :] - centroids_df[k, :]) for k in range(K)]
            for row in range(rows)
        ],
            axis=2
        )

    return distance



def assign_label(dist_matrix):

    assigned = np.argmin(
        dist_matrix,
        axis=1
    )

    return assigned



distance_matrix = distance_cal(X_train, centroids_random_k)

assigned_label_array = assign_label(distance_matrix)




# 3. Updating the centroids:
# - Once the data points are assigned to the clusters, the centroids are updated by calculating the mean of all the data points in each cluster.
# - The new centroids become the center of their respective clusters.

def mean_centroids(dataframe, assigned_result):

    assert(isinstance(assigned_result, np.ndarray))
    centeroids = np.unique(assigned_result)

    # Merge assigned centroid data to dataframe
    df = dataframe.copy()
    df['centroid'] = assigned_result

    # Calculate the mean_matrix for each centroid:
    ## return matrix:  k x cols
    means = [
        np.mean(df[df['centroid'] == center], axis=0)
        for center in centeroids
    ]

    means = pd.DataFrame(means)
    return means

means_matrix = mean_centroids(X_train, assigned_label_array)



# 4.Repeating Steps 2 and 3:
# - Steps 2 and 3 are repeated until the centroids no longer move significantly or a maximum number of iterations is reached.
# - At each iteration, the data points are reassigned to the nearest centroids, and the centroids are updated again.

def update_centroid(dataframe, centroids_mean, dist_type = 'Euclidean'):

    # Set the maximum number of iterations and convergence threshold
    max_iter = 100
    convergence_threshold = 1e-4

    wcss = 0
    centroids_df = centroids_mean.copy()

    for i in range(max_iter):

        # Recalculate the means
        distances = distance_cal(
            dataframe,
            centroids_df.drop('centroid', axis=1),
            distance_type=dist_type
        )

        centroid_array = assign_label(distances)

        updated_centroids_mean = mean_centroids(dataframe, centroid_array)

        # Determine SSE
        # Calculate the squared distances between each data point and the centroid
        updated_wcss = np.sum(
            np.array((updated_centroids_mean - centroids_mean) ** 2)
        )

        if abs(updated_wcss - wcss) < convergence_threshold:
            print(f'Converged at {i + 1} times')
            break


        wcss = updated_wcss
        centroids_df = updated_centroids_mean
        print(f'Continue after {i + 1} times...')


    return centroids_df, centroid_array


tuned_means_matrix, tuned_assigned_centroids = update_centroid(X_train, means_matrix)

def tuned_final_centroids(k, dataframe):

    centroids_random_k = centroids_initialize(k, X_train)

    distance_matrix = distance_cal(X_train, centroids_random_k)

    assigned_label_array = assign_label(distance_matrix)

    means_matrix = mean_centroids(X_train, assigned_label_array)

    tuned_means_matrix, tuned_assigned_centroids = update_centroid(X_train, means_matrix)

    return tuned_means_matrix, tuned_assigned_centroids


final_means_matrix, final_assigned_labels = tuned_final_centroids(K, X_train)


# 5. Determining the optimal number of clusters:
# - The number of clusters (K) can greatly affect the results of the K-means algorithm.
# - Therefore, it is important to determine the optimal value of K before running the algorithm.
# - The elbow method or silhouette analysis can be used to determine the optimal value of K.

def compare_k_performance(k_range, dataframe):

    wcss_dict = {}

    for k in k_range:

        print(f'Training for {k} clusters -------------')
        tuned_means_matrix, tuned_assigned_centroids = tuned_final_centroids(k, X_train)

        wcss = np.sum([
            np.sum(
                np.square(
                    X_train.iloc[
                        np.where(tuned_assigned_centroids == i)
                    ].to_numpy() \
                    - \
                    tuned_means_matrix[
                        tuned_means_matrix['centroid'] == i
                    ].drop('centroid', axis=1).to_numpy()
                )
            )
            for i in range(k)
        ])

        wcss_dict[k] = wcss

    return wcss_dict


wcss_result = compare_k_performance(
    [1, 2, 3, 4, 5, 6],
    X_train
)

plt.plot(wcss_result.keys(), wcss_result.values())
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')



# 6. Output:
#
# - The output of the K-means algorithm is the final centroids and the assignment of each data point to a cluster.
# - The results can be visualized using various plots such as scatter plots or heatmaps.
# Visualize the cluster assignments using a scatter plot
plt.scatter(X_train.loc[:, 'sepal_length'], X_train.loc[:, 'sepal_width'], c=final_assigned_labels, cmap='viridis')
plt.scatter(final_means_matrix.loc[:, 'sepal_length'], final_means_matrix.loc[:, 'sepal_width'], marker='*', s=200, c='#050505')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-means clustering on the Iris dataset')
plt.show()