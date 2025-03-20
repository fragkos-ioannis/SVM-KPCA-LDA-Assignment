import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from SVM_Assignment import apply_kpca, apply_lda

def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #X_train, y_train = X_train[:10000], y_train[:10000]
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return X_train, X_test, y_train, y_test

def load_and_preprocess_cifar100():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    X_train, y_train = X_train[:10000], y_train[:10000]
    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return X_train, y_train, X_test, y_test

def fit_knn_classifier(X_fit, y_fit, n_neighbors=5):
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_fit, y_fit)
    fitting_time = time.time() - start_time

    return knn, fitting_time

def evaluate_classifier(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, report

def main():
    # Load and preprocess data
    #X_train, X_test, y_train, y_test = load_cifar10_data()
    X_train, X_test, y_train, y_test = load_and_preprocess_cifar100()

    '''x_train_kpca, x_test_kpca = apply_kpca(X_train, X_test)
    print(f"Transformed training data shape: {x_train_kpca.shape}")
    print(f"Transformed test data shape: {x_test_kpca.shape}")'''

    '''x_train_lda, x_test_lda = apply_lda(X_train, y_train, X_test, 9)
    print(f"Transformed training data shape: {x_train_lda.shape}")
    print(f"Transformed test data shape: {x_test_lda.shape}")'''

    k_neighbors = [3, 5, 7, 9]

    for k in k_neighbors:
        print(f"\n--- Results for k={k} ---")
        # Train the KNN model
        knn, fitting_time = fit_knn_classifier(X_train, y_train, n_neighbors=k)
        print(f"Fitting Time: {fitting_time:.2f} seconds")

        # Make predictions
        start_time = time.time()
        y_pred = knn.predict(X_test)
        testing_time = time.time() - start_time
        print(f"Prediction Time: {testing_time:.2f} seconds")

        # Evaluate the model
        accuracy, precision, recall, f1, report = evaluate_classifier(y_test, y_pred)

        # Print results
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("\nClassification Report:\n", report)

# Execute the program
if __name__ == "__main__":
    main()
