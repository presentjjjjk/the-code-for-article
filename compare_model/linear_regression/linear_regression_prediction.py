import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

column = 'pc'
file_path = f'matched_compounds_data_{column}.csv'

data = pd.read_csv(file_path)

feature_columns = [col for col in data.columns if col not in ['compound_name', 'boiling_point', 'pc', 'SMILES']]
X = data[feature_columns].values
y = data[column].values

valid_indices = []
X_valid = []
y_valid = []

for i in range(len(X)):
    if not np.isnan(y[i]) and not np.any(np.isnan(X[i])):
        valid_indices.append(i)
        X_valid.append(X[i])
        y_valid.append(y[i])

X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

print(f"Total samples: {len(data)}")
print(f"Valid samples: {len(X_valid)}")
print(f"Feature dimensions: {X_valid.shape[1]}")

X_indices = np.arange(len(X_valid))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

best_r2 = -np.inf
best_model = None

for fold, (train_index, test_index) in enumerate(kf.split(X_indices)):
    
    X_temp_indices = X_indices[test_index]
    X_temp = X_valid[test_index]
    y_temp = y_valid[test_index]
    
    X_dev_indices, X_test_indices, X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp_indices, X_temp, y_temp, test_size=0.5, random_state=fold
    )
    

    X_train = X_valid[train_index]
    y_train = y_valid[train_index]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_test_predicted = model.predict(X_test)

    r2 = r2_score(y_test, y_test_predicted)
    mae = mean_absolute_error(y_test, y_test_predicted)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predicted))
    max_error = np.max(np.abs(y_test - y_test_predicted))
    mre = np.mean(np.abs((y_test - y_test_predicted) / y_test)) * 100

    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

print(f"best R² Score: {best_r2:.4f}")
print(f"average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")
