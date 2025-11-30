# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


# 1. Load dataset
dataset = pd.read_csv('games.csv')
print(dataset.head())
print(dataset.dtypes)
print(dataset.shape)

print(100 * dataset.isnull().sum() / len(dataset))

print(dataset.describe().round(2))

# 2. Drop columns with >50% missing values
missing_pct = dataset.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 50].index
print(f"Columns to drop (>50% missing): {list(cols_to_drop)}")

data = dataset.drop(columns=cols_to_drop)
data = data.dropna()
print(data.dtypes)


# 3. Prepare data
df = data.copy()

target = "Estimated owners"
X = df.drop(columns=[target])
y = df[target]

# Encode categorical variables
for col in X.select_dtypes(include=["object"]).columns:
    lbl = LabelEncoder()
    X[col] = lbl.fit_transform(X[col])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize (for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n===== Linear Regression =====")
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print("MAE:", round(mae_lr, 2))
print("MSE:", round(mse_lr, 2))
print("RMSE:", round(rmse_lr, 2))

# 5. KNN Regression
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\n===== KNN Regression =====")
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)

print("MAE:", round(mae_knn, 2))
print("MSE:", round(mse_knn, 2))
print("RMSE:", round(rmse_knn, 2))

# 6. Tuning KNN (K = 1 ... 30)
mae_list = []
mse_list = []
k_values = range(1, 31)

for k in k_values:
    knn_k = KNeighborsRegressor(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    pred_k = knn_k.predict(X_test_scaled)
    mae_list.append(mean_absolute_error(y_test, pred_k))
    mse_list.append(mean_squared_error(y_test, pred_k))

plt.figure(figsize=(10,5))
plt.plot(k_values, mae_list, marker='o')
plt.xlabel("K value")
plt.ylabel("MAE")
plt.title("KNN - MAE for different K values")
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(k_values, mse_list, marker='o')
plt.xlabel("K value")
plt.ylabel("MSE")
plt.title("KNN - MSE for different K values")
plt.grid()
plt.show()

best_k = k_values[mae_list.index(min(mae_list))]
print("\nBest K =", best_k)

# Model with best K
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn_best = knn_best.predict(X_test_scaled)

print("\n===== KNN (Best K) =====")
print("MAE:", round(mean_absolute_error(y_test, y_pred_knn_best), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred_knn_best), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_knn_best)), 2))


# 7. Tuning Polynomial Linear Regression
degrees = [1, 2, 3, 4]
mae_poly = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred_poly = model.predict(X_test_poly)
    
    mae_poly.append(mean_absolute_error(y_test, y_pred_poly))

plt.figure(figsize=(10,5))
plt.plot(degrees, mae_poly, marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("MAE")
plt.title("Polynomial Regression - MAE per degree")
plt.grid()
plt.show()

best_degree = degrees[mae_poly.index(min(mae_poly))]
print("\nBest polynomial degree =", best_degree)

# Final polynomial model
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

final_lr = LinearRegression()
final_lr.fit(X_train_poly, y_train)
y_pred_poly_final = final_lr.predict(X_test_poly)

print("\n===== Polynomial Linear Regression (Best Degree) =====")
print("MAE:", round(mean_absolute_error(y_test, y_pred_poly_final), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred_poly_final), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_poly_final)), 2))


# 8. Visualisations (Actual vs Predicted)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression - Actual vs Predicted")
plt.grid()
plt.show()

plt.scatter(y_test, y_pred_knn_best, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("KNN (Best K) - Actual vs Predicted")
plt.grid()
plt.show()

# Error distribution
errors = y_test - y_pred_knn_best
plt.hist(errors, bins=30)
plt.xlabel("Error")
plt.title("KNN - Prediction Error Distribution")
plt.grid()
plt.show()

# Residual plot
plt.scatter(y_pred_knn_best, errors, alpha=0.5)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("KNN Residual Plot")
plt.axhline(0, color='red')
plt.grid()
plt.show()
