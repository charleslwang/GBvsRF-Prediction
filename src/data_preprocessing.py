import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Clean and preprocess the data for modeling.

    Parameters:
    data (pandas.DataFrame): The raw data.

    Returns:
    pandas.DataFrame: The processed data ready for modeling.
    """
    # Fill missing values
    data.fillna(method='ffill', inplace=True)  # forward fill to handle missing values

    # Feature engineering (if necessary)
    # For example, you might want to add moving averages, RSI, etc.

    # Scale the data if using features that require scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    scaled_data = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=data.index)

    return scaled_data


