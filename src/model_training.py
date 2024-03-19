from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting model on the training data.

    Parameters:
    X_train (numpy.ndarray): Training features.
    y_train (numpy.ndarray): Training target variable.

    Returns:
    model: The trained Gradient Boosting model.
    """
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model on the training data.

    Parameters:
    X_train (numpy.ndarray): Training features.
    y_train (numpy.ndarray): Training target variable.

    Returns:
    model: The trained Random Forest model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print out performance metrics.

    Parameters:
    model: The trained model (Gradient Boosting or Random Forest).
    X_test (numpy.ndarray): Testing features.
    y_test (numpy.ndarray): Testing target variable.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Performance Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return mse, r2