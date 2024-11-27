import numpy as np

def accuracy(predictions, labels):
    return int(sum(labels == predictions) / len(labels) * 100)


def normalize(data, feature_range=(0, 1)):

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    scale = feature_range[1] - feature_range[0]
    normalized_data = (data - data_min) / (data_max -
                                           data_min) * scale + feature_range[0]
    return normalized_data, {"min": data_min, "max": data_max, "range": scale}


def denormalize(normalized_data, params, feature_range=(0, 1)):
    data_min = params["min"]
    data_max = params["max"]
    scale = params["range"]
    denormalized_data = (
        normalized_data - feature_range[0]) / scale * (data_max - data_min) + data_min
    return denormalized_data
