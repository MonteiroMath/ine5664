import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



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
    
def splitData(features, labels, validationPercentage=0.2):
    featuresTrain, featuresVal, targetTrain, targetVal = train_test_split(
        features,
        labels,
        test_size=validationPercentage,
    )
    return featuresTrain, featuresVal, targetTrain, targetVal



def evaluate_classification(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)

    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions),
        "Recall": recall_score(labels, predictions),
        "F1 Score": f1_score(labels, predictions),

    }
    return metrics


def evaluate_multiclass_with_one_hot(labelOneHot, predictionOneHot):

    # Converte one-hot labels para indices de classes
    label = np.argmax(labelOneHot, axis=1)
    prediction = np.argmax(predictionOneHot, axis=1)

    # Computa as métricas
    metrics = {
        'accuracy': accuracy_score(label, prediction),
        'precision': precision_score(label, prediction, average='weighted', zero_division=0),
    }

    print("Evaluation Metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


    
