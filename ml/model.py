import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Computes precision, recall, and F1 (fbeta) metrics.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inference.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Saves a model to a file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Loads a model from a file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Computes model performance metrics on a slice of the data.

    Parameters:
        data: full dataset (DataFrame)
        column_name: column to slice on
        slice_value: value within the column to filter on
        categorical_features: list of categorical feature names
        label: label column name
        encoder: fitted OneHotEncoder
        lb: fitted LabelBinarizer
        model: trained model

    Returns:
        precision, recall, fbeta for the slice
    """
    # Filter the slice
    data_slice = data[data[column_name] == slice_value]

    # If slice is empty, avoid crashing
    if data_slice.empty:
        return None, None, None

    # Process slice
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict
    preds = inference(model, X_slice)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
