from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import time
import pandas as pd

# Function to load and split the California housing dataset
def load_housing_data():
    data = fetch_california_housing(as_frame=True)
    X = data.data  # California housing features
    y = data.target  # Median house value

    print("Columns of X (features):", X.columns.tolist())
    print("\nFirst 5 rows of X:\n", X.head())
    print("\nFirst 5 values of y:\n", y.head())
    
    return train_test_split(X, y, test_size=0.4, random_state=42)


def load_movie_data(file_path):
    
    # Load the dataset
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")
    print("Preview of the data:\n", data.head())
    print("Dataset info:\n")
    data.info()

    # Define features and target
    features = ['budget', 'runtime', 'popularity', 'vote_average', 'vote_count']
    target = 'revenue'

    # Drop rows with missing values
    df = data[features + [target]].dropna()

    # Split features and target
    X = df[features][:100000]
    y = df[target][:100000]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Display preview of data splits
    print("\nHead of X_train (before scaling):\n", X_train.head())
    print("\nHead of X_test (before scaling):\n", X_test.head())
    print("\nHead of y_train:\n", y_train.head())
    print("\nHead of y_test:\n", y_test.head())

    # Display number of samples in each split
    print(f"\nNumber of training samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test



# Function to preprocess data by standardizing features
def preprocess_data(X_train, X_test, y_train, y_test, scale_target=False):
    scaler_X = RobustScaler()
    #scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    if scale_target:
        #scaler_y = RobustScaler()
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y
    else:
        return X_train_scaled, X_test_scaled, y_train, y_test, None

# Function to train an SVR model and report training score
def train_svr(X_train, y_train, C=1.0, epsilon=0.2, kernel='rbf'):
    svr = SVR(C=C, epsilon=epsilon, kernel=kernel)
    start_time = time.time()  # Start timing
    svr.fit(X_train, y_train)
    training_time = time.time() - start_time  # Calculate training time

    y_train_pred = svr.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    
    print(f"Training MSE: {mse_train:.4f}")
    print(f"Training R-squared: {r2_train:.4f}")

    return svr, training_time

# Function to evaluate the model performance on test data
def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    testing_time = time.time() - start_time
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test R-squared: {r2}")
    return mse, r2, y_pred, testing_time

# Function to perform hyperparameter tuning with GridSearchCV and display scores
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'C': [5, 10, 15, 20],
        'epsilon': [0.1, 0.2, 0.4, 0.6],
        'kernel': ['rbf']
    }
    svr = SVR()

    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=3,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    # Extract and display training and validation scores for each parameter combination
    results = grid_search.cv_results_
    print("\nHyperparameter Tuning Results:")
    for mean_train_score, mean_test_score, params, mean_fit_time in zip(
        results["mean_train_score"], results["mean_test_score"], results["params"], results["mean_fit_time"]
    ):
        print(f"Params: {params}")
        print(f"Mean Training MSE: {-mean_train_score:.4f}")  # Negate as MSE is negative in scoring
        print(f"Mean Validation MSE: {-mean_test_score:.4f}")
        print(f"Time Taken: {mean_fit_time:.4f} seconds\n")
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score (Validation MSE):", -grid_search.best_score_)
    
    return grid_search.best_estimator_, results

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m and {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"