import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr=0.1, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None  # Initialize the weights attribute to None
        self.bias = None
        self.loss_history = []

        
    def softmax(self, z):
        """
        Compute the softmax of the input logits z.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cost(self, h, y):
        """
        Compute the cross-entropy loss between the predicted
        probability distribution h and the true distribution y.
        """
        return (-y * np.log(h)).mean()

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
        self.loss_history = []  # Initialize the loss_history attribute

        for _ in range(int(self.max_iters)):

            z = np.dot(training_data, self.weights) + self.bias
            h = self.softmax(z)
            loss = self.cost(h, y_onehot)
            self.loss_history.append(loss)  # Append the loss to the loss_history attribute
            
            gradient_w = np.dot(training_data.T, (h - y_onehot)) / n_samples
            gradient_b = np.sum(h - y_onehot, axis=0) / n_samples

            self.weights -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b
            
        return self

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        z = np.dot(test_data, self.weights) + self.bias
        h = self.softmax(z)
        pred_onehot = (h == h.max(axis=1, keepdims=1)).astype(int)
        pred_labels = onehot_to_label(pred_onehot)
        return pred_labels

    
    def set_params(self, **params):
        self.lr = float(params.get('lr', self.lr))
        self.max_iters = int(params.get('max_iters', self.max_iters))
        self.weights = params.get('weights', self.weights)
        self.bias = params.get('bias', self.bias)

    def score(self, X, y):
        """
        Runs prediction on the test data and returns the mean accuracy.

        Arguments:
            X (array): test data of shape (N,D)
            y (array): labels of shape (N,)
        Returns:
            mean accuracy (float)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
