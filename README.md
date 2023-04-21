# Handwritten-Math-Symbol-Recognition
A machine learning project that focuses on recognizing handwritten mathematical symbols using various classification algorithms like K-Means, Logistic Regression, and SVM. This repository includes a step-by-step implementation, evaluation, and comparison of the methods using the HASYv2 dataset, a small dataset of hand-drawn mathematical symbols.


# Requirements
Python 3
NumPy
Matplotlib
Scikit-learn

# Installation

* Clone the repository:

`git clone https://github.com/SkanderGasmi/Handwritten-Math-Symbol-Recognition.git`

* Navigate to the project directory:
`cd Handwritten-Math-Symbol-Recognition`

* Create a virtual environment :
`python -m venv venv`

* Activate the virtual environment:

`venv\Scripts\activate  # For Windows
source venv/bin/activate   # For Linux/Mac `

* Install the required dependencies:

`pip install -r requirements.txt`

# Usage

Download the HASYv2 dataset from Moodle and place it in the project directory.

Run the main script with the desired arguments:

python main.py --data <path_to_data_folder> --method <kmeans|logistic_regression|svm> --<hyperparameter_name> <value>

For example, to run K-Means with K=3, use the following command:

python main.py --data data --method kmeans --K 3

You can also specify additional arguments such as the learning rate and max iterations for logistic regression, or the kernel type and gamma for SVM. Check the main.py script for all available arguments.

After running the main script, the results will be printed in the console.

You can also run the test script to verify your project folder and code structure:

python test_ms1.py

Make sure that the test script runs without any problems.

Finally, you can generate a report by filling out the relevant sections in the report template provided in the project folder.

report.pdf

The report should be a concise 2-page document that describes your methodology, hyperparameter selection process, and the performance of the implemented methods on the dataset.
