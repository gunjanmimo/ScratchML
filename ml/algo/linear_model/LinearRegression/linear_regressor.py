# %%
import numpy as np
from typing import List
from tqdm import tqdm


class LinearRegressor:
    def __init__(self) -> None:
        self.weights = None
        self.bias = None

    def fit(self, x: np.ndarray, y: np.ndarray, n_iters=1000, lr: float = 0.001):
        n_samples, n_features = x.shape
        # init weight and bias value
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in tqdm(range(n_iters)):
            y_pred = self.predict(x=x)
            dw = -(2 * (x.T).dot(y - y_pred)) / n_samples
            db = -2 * np.sum(y - y_pred) / n_samples

            self.weights = self.weights - lr * dw
            self.bias = self.bias - lr * db

    def predict(self, x: np.ndarray) -> float:
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred

    def coefficient(self) -> List[float]:
        return {"weights": self.weights, "bias": self.bias}

    def export(self) -> bool:
        return

    def load(self) -> bool:
        return


# %%
import pandas as pd

if __name__ == "__main__":
    print("-------------------------")
    data = [
        {
            "val_1": i,
            #  "val_2": j,
            "result": i + 100,
        }
        for i in range(1, 10)
        # for j in range(1, 9)
    ]
    data = pd.DataFrame(data)
    X = data[["val_1"]].values
    y = data[["result"]].values

    # model
    model = LinearRegressor()
    model.fit(x=X, y=y)
    print(model.weights)
    print(model.predict(np.array([[2]])))

# %%
