import pandas as pd
import numpy as np
from thermo import Joback
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

column = 'Vc'
file_path = f'matched_compounds_data_{column}.csv'

data = pd.read_csv(file_path)

def predict_boiling_point_joback(smiles):
    try:
        J = Joback(smiles)
        tb = J.Vc(J.counts) 
        return tb
    except Exception:
        return np.nan

predicted_boiling_points = []
valid_indices = []

for i, smiles in enumerate(data['SMILES']):
    predicted_bp = predict_boiling_point_joback(smiles)
    if not np.isnan(predicted_bp):
        predicted_boiling_points.append(predicted_bp)
        valid_indices.append(i)
    else:
        pass


valid_data = data.iloc[valid_indices].copy()
valid_data['predicted_bp'] = predicted_boiling_points

X_indices = np.arange(len(valid_data))
y_actual = valid_data[column].values*1000
y_predicted = valid_data['predicted_bp'].values * 1e6

kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

best_r2 = -np.inf

for fold, (train_index, test_index) in enumerate(kf.split(X_indices)):

    X_temp_indices = X_indices[test_index]
    y_temp_actual = y_actual[test_index]
    y_temp_predicted = y_predicted[test_index]
    
    X_dev_indices, X_test_indices, y_dev_actual, y_test_actual, y_dev_predicted, y_test_predicted = train_test_split(
        X_temp_indices, y_temp_actual, y_temp_predicted, test_size=0.5, random_state=fold
    )
    
    r2 = r2_score(y_test_actual, y_test_predicted)
    mae = mean_absolute_error(y_test_actual, y_test_predicted)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_predicted))
    max_error = np.max(np.abs(y_test_actual - y_test_predicted))
    mre = np.mean(np.abs((y_test_actual - y_test_predicted) / y_test_actual)) * 100

    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)

    if r2 > best_r2:
        best_r2 = r2

print(f"best R² Score: {best_r2:.4f}")
print(f"average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")

