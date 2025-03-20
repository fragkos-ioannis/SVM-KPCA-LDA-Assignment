# SVM with Kernel PCA and LDA for CIFAR-10 and CIFAR-100

This project applies Support Vector Machines (SVM) along with Kernel Principal Component Analysis (Kernel PCA) and Linear Discriminant Analysis (LDA) to classify images from the CIFAR-10 and CIFAR-100 datasets. The code is designed to load and preprocess data, apply dimensionality reduction techniques, tune the SVM hyperparameters, and evaluate model performance.

## Project Overview

The goal of this project is to classify images from the CIFAR-10 and CIFAR-100 datasets using SVMs. Kernel PCA is used for dimensionality reduction before training the model, while LDA is also available as an alternative preprocessing step. Hyperparameter tuning is performed to optimize SVM performance, and the model is evaluated using accuracy, confusion matrix, and classification report.

## Dependencies

To run this project, you need the following Python packages:
- `numpy`
- `matplotlib`
- `tensorflow`
- `sklearn`

You can install the required libraries using `pip`:
pip install numpy matplotlib tensorflow scikit-learn


## Key Functions

### `load_and_preprocess_data()`
Loads and preprocesses the CIFAR-100 dataset by normalizing pixel values and flattening the images for classification.

### `apply_kpca(x_train, x_test)`
Applies Kernel PCA for dimensionality reduction to the training and test data. It also visualizes the eigenvalues and cumulative explained variance.

### `apply_lda(x_train, y_train, x_test)`
Applies Linear Discriminant Analysis (LDA) to the data (not used by default in this script).

### `train_svm(x_train, y_train)`
Trains an SVM model with the provided training data.

### `tune_svm_hyperparameters(x_train, y_train)`
Tunes hyperparameters for the SVM model using `GridSearchCV` to find the optimal `C` parameter. It evaluates the model with 5-fold cross-validation.

### `test_svm(clf, x_test)`
Evaluates the trained SVM model on the test data, returning the predicted values and testing time.

### `evaluate(y_test, y_pred)`
Evaluates the model’s performance by calculating accuracy and generating a classification report and confusion matrix.

## How to Run

1. Clone this repository to your local machine:
git clone https://github.com/fragkos-ioannis/SVM-KPCA-LDA-Assignment.git


2. Navigate to the project directory:
cd SVM-KPCA-LDA-Assignment


3. Run the main script:
python svm_kpca_lda.py


## Results

The model's performance will be displayed, including:
- Training time
- Training accuracy
- Testing time
- Testing accuracy
- Classification report
- Confusion matrix visualization

## Potential Improvements

- Implement LDA for dimensionality reduction as a default method
- Experiment with different kernels for the SVM model
- Tune additional hyperparameters like `gamma` for the RBF kernel

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


