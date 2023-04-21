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

bash
Copy code
cd Handwritten-Math-Symbol-Recognition
Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

bash
Copy code
source venv/Scripts/activate  # For Windows
source venv/bin/activate     # For Linux/Mac
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Navigate to the project directory:

bash
Copy code
cd Handwritten-Math-Symbol-Recognition
Activate the virtual environment:

bash
Copy code
source venv/Scripts/activate  # For Windows
source venv/bin/activate     # For Linux/Mac
Run the desired script from the src directory:

bash
Copy code
python src/k_means.py  # For K-Means
python src/logistic_regression.py  # For Logistic Regression
python src/svm.py  # For SVM
Results
The results of the different classification methods are presented in the results directory. The evaluation metrics and the trained models can be found in the respective subdirectories for each method.

Dataset
The HASYv2 dataset used in this project can be downloaded from here. The dataset is included in the data directory and is split into training, validation, and testing sets.

License
This project is licensed under the MIT License - see the LICENSE file for details.
