import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from SVR_Assignment import load_movie_data, preprocess_data, format_time

def fit_knn_regressor(X_fit, y_fit, n_neighbors=5):
    """
    Trains a KNN Regressor on the training data.
    """
    start_time = time.time()
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_fit, y_fit)
    fiting_time = time.time() - start_time

    return knn, fiting_time

def evaluate_model(y_test, y_pred):
    """
    Evaluates the model using Mean Squared Error and Mean Absolute Error.
    """
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def main():

    # Load and preprocess data
    file_path = "./movieData/TMDB_movie_dataset_v11.csv"
    X_fit, X_test, y_fit, y_test = load_movie_data(file_path)

    # Preprocess
    X_fit_scaled, X_test_scaled, y_fit_scaled, y_test_scaled, scaler_y = preprocess_data(
        X_fit, X_test, y_fit, y_test, scale_target=True
    )

    k_neighbors = [3, 5, 7] #[8, 10, 12]

    for k in k_neighbors:
        print(print(f"\n--- Results for k={k} ---"))
        # Train the KNN model
        knn, fiting_time = fit_knn_regressor(X_fit_scaled, y_fit_scaled, n_neighbors=k)
        print(f"Fiting Time: {format_time(fiting_time)}")

        # Make predictions
        start_time = time.time()
        y_pred = knn.predict(X_test_scaled)
        testing_time = time.time() - start_time
        print(f"Prediction Time: {format_time(testing_time)}")

        # Evaluate the model
        mse, mae, r2 = evaluate_model(y_test_scaled, y_pred)

        # Print results
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print(f"R-squared: {r2}")

# Execute the program
if __name__ == "__main__":
    main()
