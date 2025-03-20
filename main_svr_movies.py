from SVR_Assignment import load_movie_data, format_time, tune_hyperparameters, preprocess_data, train_svr, evaluate_model

def main():
    # Load the dataset
    file_path = "./movieData/TMDB_movie_dataset_v11.csv"
    X_train, X_test, y_train, y_test = load_movie_data(file_path)

    print("train size ", X_train, "test_size: ", X_test)
    # Preprocess data by scaling features
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y = preprocess_data(
        X_train, X_test, y_train, y_test, scale_target=True
    )

    # Tune hyperparameters using GridSearchCV
    #best_svr_model, results = tune_hyperparameters(X_train_scaled, y_train_scaled)

    # Retrieve best parameters from the tuned model
    #best_params = best_svr_model.get_params()
    
    # Train model with the best hyperparameters
    best_model, training_time = train_svr(
        X_train_scaled, y_train_scaled, 
        C=8, 
        epsilon=0.7, 
        kernel='linear'
    )
    print(f"Training Time: {format_time(training_time)}")

    # Evaluate the trained model on the test set
    mse_test, r2_test, y_pred, testing_time = evaluate_model(best_model, X_test_scaled, y_test_scaled)
    print(f"Testing Time: {format_time(testing_time)}")

    if scaler_y is not None:
        y_pred_original_scale = scaler_y.inverse_transform(best_model.predict(X_test_scaled).reshape(-1, 1)).flatten()
        y_test_original_scale = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        print("\nPredictions and test targets in original scale:")
        print("Predictions:", y_pred_original_scale[:5])
        print("Test targets:", y_test_original_scale[:5])

if __name__ == "__main__":
    main()