from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
import time
from SVM_Assignment import apply_lda, apply_kpca

def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, y_train = X_train[:60000], y_train[:60000]
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Normalize pixel values to [0, 1]
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return X_train, X_test, y_train, y_test
    

def fit_ncc_classifier(X_fit, y_fit):
    start_time = time.time()
    ncc = NearestCentroid()
    ncc.fit(X_fit, y_fit)
    fitting_time = time.time() - start_time
    return ncc, fitting_time

def evaluate_classifier(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, report

def main():
    # Load and preprocess CIFAR-10 data
    X_train, X_test, y_train, y_test = load_cifar10_data()

    x_train_kpca, x_test_kpca = apply_kpca(X_train, X_test)
    print(f"Transformed training data shape: {x_train_kpca.shape}")
    print(f"Transformed test data shape: {x_test_kpca.shape}")

    x_train_lda, x_test_lda = apply_lda(x_train_kpca, y_train, x_test_kpca, 8)
    print(f"Transformed training data shape: {x_train_lda.shape}")
    print(f"Transformed test data shape: {x_test_lda.shape}")

    # Train the NCC model
    print("\n--- Fitting Nearest Centroid Classifier ---")
    ncc, fitting_time = fit_ncc_classifier(x_train_lda, y_train)
    print(f"Fitting Time: {fitting_time:.2f} seconds")

    # Make predictions
    print("\n--- Evaluating Nearest Centroid Classifier ---")
    start_time = time.time()
    y_pred = ncc.predict(x_test_lda)
    prediction_time = time.time() - start_time
    print(f"Prediction Time: {prediction_time:.2f} seconds")

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
