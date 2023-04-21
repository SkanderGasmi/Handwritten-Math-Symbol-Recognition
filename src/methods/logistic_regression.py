import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        n_samples, n_features = training_data.shape
        n_classes = get_n_classes(training_labels)
        y_onehot = label_to_onehot(training_labels, n_classes)

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for _ in range(self.max_iters):
            z = np.dot(training_data, self.weights) + self.bias
            h = self.sigmoid(z)
            
            gradient_w = np.dot(training_data.T, (h - y_onehot)) / n_samples
            gradient_b = np.sum(h - y_onehot, axis=0) / n_samples

            self.weights -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b
            
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        z = np.dot(test_data, self.weights) + self.bias
        h = self.sigmoid(z)
        pred_onehot = (h == h.max(axis=1, keepdims=1)).astype(int)
        pred_labels = onehot_to_label(pred_onehot)
        return pred_labels