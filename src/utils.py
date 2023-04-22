import numpy as np 
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from tqdm import tqdm



# Generaly utilies
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)

def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]),data], axis=1)
    return data

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds

def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
 
def grid_search(model, param_grid, X, y, cv=5, scoring='accuracy'):
    # Generate all possible combinations of hyperparameter values
    param_combinations = np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))

    # Create an array to hold the validation scores for each combination of hyperparameters
    scores = np.zeros(param_combinations.shape[0])

    # Create a progress bar
    

    progress_bar = tqdm(total=param_combinations.shape[0], desc="Grid search progress",
                    bar_format="{l_bar}\u001b[41m{bar:40}\u001b[0m {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]\n",
                    ncols=50)

    # Perform k-fold cross-validation on each combination of hyperparameters
    n_samples = X.shape[0]
    fold_size = n_samples // cv

    for i, params in enumerate(param_combinations):
        score_sum = 0.0

        for j in range(cv):
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[j * fold_size:(j + 1) * fold_size] = True
            train_mask = np.logical_not(val_mask)

            model.set_params(**dict(zip(param_grid.keys(), params)))
            model.fit(X[train_mask], y[train_mask])
            score = model.score(X[val_mask], y[val_mask])
            score_sum += score

        scores[i] = score_sum / cv

        # Update the progress bar every 1%
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Determine the index of the combination with the highest score
    best_idx = np.argmax(scores)

    # Return the best hyperparameters and score
    best_params = dict(zip(param_grid.keys(), param_combinations[best_idx]))
    best_score = scores[best_idx]

    return best_params, best_score
