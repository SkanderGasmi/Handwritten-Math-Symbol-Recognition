import argparse

import numpy as np 
import torch
from torch.utils.data import DataLoader

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import grid_search, normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objs as go
import plotly.graph_objs as go
from IPython.display import display








def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    #xtrain = xtrain.reshape(xtrain.shape[0], -1)
    #xtest = xtest.reshape(xtest.shape[0], -1)
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1)
    xtest = xtest.reshape(xtest.shape[0], xtrain.shape[1], xtrain.shape[2], 1)

    ## Perform data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Number of augmented samples per original sample
    augmentation_factor = 2

    datagen.fit(xtrain)
    
    xtrain_augmented_list = []
    ytrain_augmented_list = []


    for x_batch, y_batch in datagen.flow(xtrain, ytrain, batch_size=xtrain.shape[0], shuffle=False):
        xtrain_augmented_list.append(x_batch)
        ytrain_augmented_list.append(y_batch)  # Update the labels list

        if len(xtrain_augmented_list) >= augmentation_factor:
            break

    xtrain_augmented = np.vstack(xtrain_augmented_list)
    ytrain_augmented = np.vstack(ytrain_augmented_list)  # Stack the labels list

    print("Shape of xtrain before data augmentation:", xtrain.shape)
    print("Shape of ytrain before data augmentation:", ytrain.shape)
    
    


    print("Shape of xtrain after data augmentation:", xtrain_augmented.shape)
    
    xtrain = xtrain_augmented.reshape(xtrain_augmented.shape[0], -1)
    ytrain = ytrain_augmented.ravel()

    print("Shape of xtrain after data augmentation and reshaping :", xtrain.shape)
    print("Shape of ytrain after data augmentation:", ytrain.shape)





    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.
    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)
   
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)

        pass
    
    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        raise NotImplementedError("This will be useful for MS2.")
    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)

    elif args.method == "svm":
        method_obj = SVM(args.svm_c, args.svm_kernel, args.svm_gamma, args.svm_degree, args.svm_coef0)

    elif args.method == "kmeans":
        method_obj = KMeans(args.K, args.max_iters)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(args.lr, args.max_iters)
        

    if args.grid_search:
        # Perform grid search for hyperparameter tuning

        # Define the hyperparameter search space for each method
        param_grids = {
            'kmeans': {
                'K': list(range(2, 21)),
                'max_iters': [100, 200, 300],
            },
            'logistic_regression': {
                'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'max_iters': [100, 200, 300, 500, 1000]
            },
            'svm': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'] + [0.1, 1, 10],
                'degree': [2, 3, 4],
                'coef0': [0.0, 1.0, 2.0],
            }
        }

        # Choose the appropriate grid for the selected method
        param_grid = param_grids[args.method]

        # Perform the grid search
        best_params, best_score = grid_search(method_obj, param_grid, xtrain, ytrain, cv=5, scoring='accuracy')

        # Print the best parameters and score
        print("Best parameters:", best_params)
        print("Best score:", best_score)
        

        # Train the classifier with the best parameters
        method_obj.__init__(**best_params)

        #method_obj.set_params(**best_params)
        method_obj.fit(xtrain, ytrain)


        # Train the classifier with the best parameters
        
    else:
        # Train and evaluate the classifier
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on validation/test data
        preds_valid = method_obj.predict(xval) if not args.test else method_obj.predict(xtest)


    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds_valid, ytest)
    macrof1 = macrof1_fn(preds_valid, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.



    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")
    parser.add_argument('--grid_search', action='store_true', help="Enable grid search for hyperparameter tuning")
    
    # Feel free to add more arguments here if you need!

    # Arguments for MS2
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)