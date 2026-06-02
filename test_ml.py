import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    data = pd.DataFrame({
        "age": [25, 45, 35, 50],
        "workclass": ["Private", "Self-emp", "Private", "Private"],
        "education": ["Bachelors", "HS-grad", "HS-grad", "Masters"],
        "marital-status": ["Never-married", "Married", "Divorced", "Married"],
        "occupation": ["Tech", "Sales", "Tech", "Exec"],
        "relationship": ["Not-in-family", "Husband", "Unmarried", "Husband"],
        "race": ["White", "Black", "White", "White"],
        "sex": ["Male", "Female", "Female", "Male"],
        "native-country": ["United-States", "United-States", "Canada", "United-States"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"],
    })
    return data


@pytest.fixture
def processed(sample_data):
    """Process the sample data."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    train, _ = train_test_split(sample_data, test_size=0.5, random_state=42)

    X, y, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    return X, y, encoder, lb, cat_features, train


# -----------------------------
# Tests
# -----------------------------
def test_train_model(processed):
    """Test that the model trains and returns a fitted model."""
    X, y, _, _, _, _ = processed
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, "predict")


def test_compute_model_metrics(processed):
    """Test that metrics return valid numeric values."""
    X, y, _, _, _, _ = processed
    model = train_model(X, y)
    preds = inference(model, X)

    p, r, fb = compute_model_metrics(y, preds)

    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= fb <= 1


def test_performance_on_categorical_slice(processed):
    """Test slice performance returns valid metrics."""
    X, y, encoder, lb, cat_features, data = processed
    model = train_model(X, y)

    col = cat_features[0]
    slice_value = data[col].iloc[0]

    p, r, fb = performance_on_categorical_slice(
        data,
        col,
        slice_value,
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        model=model
    )

    assert p is not None
    assert r is not None
    assert fb is not None
