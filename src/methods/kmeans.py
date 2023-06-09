import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        N, D = data.shape
        centers = data[np.random.choice(N, self.K, replace=False)]

        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
            cluster_assignments = np.argmin(distances, axis=1)

            new_centers = np.array([data[cluster_assignments == k].mean(axis=0) for k in range(self.K)])
            if np.allclose(centers, new_centers):
                break

            centers = new_centers
            
        return centers, cluster_assignments
    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        self.centers, self.cluster_assignments = self.k_means(training_data, self.max_iters)
        self.cluster_labels = np.zeros(self.K)

        for k in range(self.K):
            self.cluster_labels[k] = np.bincount(training_labels[self.cluster_assignments == k]).argmax()

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centers, axis=2)
        closest_cluster = np.argmin(distances, axis=1)
        pred_labels = self.cluster_labels[closest_cluster]
        
        return pred_labels
    
    def score(self, test_data, test_labels):
        """
        Calculate the accuracy of the KMeans algorithm.

        Arguments:
            test_data (np.array): test data of shape (N,D)
            test_labels (np.array): test labels of shape (N,)
        Returns:
            accuracy (float): the accuracy of the KMeans algorithm.
        """
        predicted_labels = self.predict(test_data)
        accuracy = np.mean(predicted_labels == test_labels)
        return accuracy





    
    def set_params(self, **params):
        self.K = int(params.get('K', self.K))
        self.max_iters = int(params.get('max_iters', self.max_iters))

