import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m and {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = x_train[:10000], y_train[:10000]
    # flatten images 
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    # normalize pixel values     
    x_train = x_train.astype('float32') / 255.0        
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

def load_and_preprocess_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train, y_train = x_train[:10000], y_train[:10000]
    # Flatten images
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

def apply_kpca(x_train, x_test, kernel='rbf', n_components=10000):
    
    print(f"Applying Kernel PCA with kernel='{kernel}' and n_components={n_components}...")
    kpca = KernelPCA(kernel=kernel, n_components=n_components)
    x_train_kpca = kpca.fit_transform(x_train)
    x_test_kpca = kpca.transform(x_test)
    eigenvalues = kpca.eigenvalues_
    # Calculate the total explained variance by the first n_components
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', color='b')
    plt.title('Eigenvalues from Kernel PCA')
    plt.xlabel('Component Index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    # Step 3: Compute cumulative sum of eigenvalues
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Step 4: Plot cumulative variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Sum of Kernel PCA Eigenvalues')
    plt.legend()
    plt.show()

    # Step 5: Find the number of components for 90% variance
    k_99 = np.argmax(cumulative_variance >= 0.99) + 1
    print(f"Number of components to retain 99% of the information: {k_99}")

    return x_train_kpca, x_test_kpca

def apply_lda(x_train, y_train, x_test, n_components=None):
    print(f"Applying LDA with n_components={n_components}...")
    lda = LDA(n_components=n_components)
    x_train_lda = lda.fit_transform(x_train, y_train.ravel())
    x_test_lda = lda.transform(x_test)
    print("LDA applied successfully.")
    return x_train_lda, x_test_lda


def train_svm(x_train, y_train, kernel='linear'):
    clf = svm.SVC(kernel=kernel, C=1)
    start_time = time.time()  # Start timing
    clf.fit(x_train, y_train.ravel())  # Train the model
    training_time = time.time() - start_time  # Calculate training time
    train_accuracy = accuracy_score(y_train, clf.predict(x_train))  # Calculate training accuracy
    return clf, training_time, train_accuracy


def tune_svm_hyperparameters(x_train, y_train):
    kernels = ['rbf']
    results = []

    for kernel in kernels:
        print(f"Tuning hyperparameters for kernel: {kernel}")
        param_grid = {
            'C': [0.1, 1, 10],
            #'gamma': [0.01, 0.1, 'scale'],
            'kernel': [kernel]
        }

        grid_search = GridSearchCV(
            estimator=svm.SVC(),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            verbose=3,
            return_train_score=True
        )

        start_time = time.time()
        grid_search.fit(x_train, y_train.ravel())
        cross_val_time = time.time() - start_time

        results_dict = grid_search.cv_results_

        # Calculate total fit time (training time) and total score time (prediction time) across all 5 folds
        total_fit_time = np.sum(results_dict["mean_fit_time"])  # Sum of training times
        total_score_time = np.sum(results_dict["mean_score_time"])  # Sum of prediction times

        # Calculate the mean time for both fit and score across all folds
        mean_fit_time_all_folds = total_fit_time / 5  # Divide by number of folds (5)
        mean_score_time_all_folds = total_score_time / 5  # Divide by number of folds (5)

        print("\nHyperparameter Tuning Results:")
        for mean_train_score, mean_test_score, params in zip(
            results_dict["mean_train_score"],
            results_dict["mean_test_score"],
            results_dict["params"]
        ):
            print(f"Params: {params}")
            print(f"Mean Training Accuracy: {mean_train_score:.4f}")
            print(f"Mean Validation Accuracy: {mean_test_score:.4f}")
            print(f"Mean Fit Time (Training): {mean_fit_time_all_folds:.4f} seconds")
            print(f"Mean Score Time (Prediction): {mean_score_time_all_folds:.4f} seconds\n")

        best_params = grid_search.best_params_
        best_accuracy = grid_search.best_score_

        results.append({
            'kernel': kernel,
            'best_params': best_params,
            'cross_val_accuracy': best_accuracy,
            'cross_val_time': cross_val_time,
            'mean_fit_time_all_folds': mean_fit_time_all_folds,
            'mean_score_time_all_folds': mean_score_time_all_folds
        })

        print(f"Kernel: {kernel}")
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation Accuracy: {best_accuracy:.4f}")
        print(f"Total Cross-Validation Time: {cross_val_time:.4f} seconds")
        print(f"Mean Fit Time (Training): {mean_fit_time_all_folds:.4f} seconds")
        print(f"Mean Score Time (Prediction): {mean_score_time_all_folds:.4f} seconds\n")

    return results



def test_svm(clf, x_test):
    start_time = time.time()  # Start timing
    y_pred = clf.predict(x_test)  # Predict on test set
    testing_time = time.time() - start_time  # Calculate testing time
    return y_pred, testing_time

# Function to evaluate the predictions
def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f"Testing Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    #x_train, y_train, x_test, y_test = load_and_preprocess_data()

    x_train, y_train, x_test, y_test = load_and_preprocess_cifar100()

    print(f"Number of training samples: {x_train.shape}")
    print(f"Number of test samples: {x_test.shape}")

    x_train_kpca, x_test_kpca = apply_kpca(x_train, x_test)
    #x_train_lda, x_test_lda = apply_lda(x_train, y_train, x_test, 99)

    print(f"Transformed training data shape: {x_train_kpca.shape}")
    print(f"Transformed test data shape: {x_test_kpca.shape}")

    #print(f"Transformed training data shape: {x_train_lda.shape}")
    #print(f"Transformed test data shape: {x_test_lda.shape}")

    '''results = tune_svm_hyperparameters(x_train_lda, y_train)
    print(results)'''

    # Train the SVM
    print("Training the SVM model...")
    clf, training_time, train_accuracy = train_svm(x_train_kpca, y_train)
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Training Accuracy: {train_accuracy:.4f}\n")

    # Test the SVM
    print("Testing the SVM model...")
    y_pred, testing_time = test_svm(clf, x_test_kpca)
    print(f"Testing Time: {testing_time:.4f} seconds\n")

    # Evaluate the results
    print("Evaluating the model...")
    evaluate(y_test, y_pred)


if __name__ == "__main__":
    main()





