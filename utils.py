import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def accuracy(predictions, labels):
    return int(sum(labels == predictions) / len(labels) * 100)

def normalize(data, feature_range=(0, 1)):

    data = np.array(data)
    scaler = MinMaxScaler(feature_range=feature_range)
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler


def denormalize(normalized_data, scaler):

    normalized_data = np.array(normalized_data)
    denormalized_data = scaler.inverse_transform(normalized_data)

    return denormalized_data

def evaluate_regression(targets, predictions):

    targets = np.array(targets)
    predictions = np.array(predictions)

    # Compute metrics
    mse = mean_squared_error(targets, predictions)  # Mean Squared Error
    rmse = np.sqrt(mse)                    # Root Mean Squared Error
    r2 = r2_score(targets, predictions)  # R^2 Score

    # Return metrics in a dictionary
    return {"RMSE": rmse, "R^2": r2}
    
