import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
