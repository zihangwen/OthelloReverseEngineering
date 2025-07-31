# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

plt.scatter(X, y, color="red", label="Data")
plt.title("Synthetic Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

# %%
y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%
regressor = DecisionTreeRegressor(max_depth=4, random_state=42)

# %%
regressor.fit(X_train, y_train)

# %%
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# %%
X_grid = np.arange(min(X), max(X), 0.01)[:, np.newaxis]
y_grid_pred = regressor.predict(X_grid)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="red", label="Data")
plt.plot(X_grid, y_grid_pred, color="blue", label="Model Prediction")
plt.title("Decision Tree Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

# %%
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(regressor, feature_names=["Feature"], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.show()