import time
from day5.演習2.main import DataLoader, ModelTester


def test_model_inference_accuracy():
    model = ModelTester.load_model("day5/演習2/main.py")
    data = DataLoader.load_titanic_data("day5/演習1/data/Titanic.csv")
    X, y = DataLoader.preprocess_titanic_data(data)
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    assert accuracy >= 0.75, f"Accuracy too low: {accuracy:.4f}"


def test_model_inference_time():
    model = ModelTester.load_model("day5/演習2/main.py")
    data = DataLoader.load_titanic_data("day5/演習1/data/Titanic.csv")
    X, y = DataLoader.preprocess_titanic_data(data)
    start = time.time()
    _ = model.predict(X)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Inference took too long: {elapsed:.4f} sec"
