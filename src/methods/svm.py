"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC
from sklearn.svm import SVC


class SVM(object):
    """
    SVM method.
    """

    def __init__(self, C, kernel, gamma=1., degree=1, coef0=0.):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            C (float): the weight of penalty term for misclassifications
            kernel (str): kernel in SVM method, can be 'linear', 'rbf' or 'poly' (:=polynomial)
            gamma (float): gamma prameter in rbf and polynomial SVM method
            degree (int): degree in polynomial SVM method
            coef0 (float): coef0 in polynomial SVM method
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.svc = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)

        
    def fit(self, training_data, training_labels):
        """
        Trains the model by SVM, then returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.svc.fit(training_data, training_labels)
        return self.predict(training_data)
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = self.svc.predict(test_data)

        return pred_labels
    
    
    
    def set_params(self, **params):
        self.C = float(params.get('C', self.C))
        self.kernel = params.get('kernel', self.kernel)
        self.gamma = params.get('gamma', self.gamma)
        self.degree = int(params.get('degree', self.degree))
        self.coef0 = float(params.get('coef0', self.coef0))
        self.svc.set_params(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)


    def score(self, X, y):
        """
        Runs prediction on the test data and returns the mean accuracy.

        Arguments:
            X (array): test data of shape (N,D)
            y (array): labels of shape (N,)
        Returns:
            mean accuracy (float)
        """
        return self.svc.score(X, y)