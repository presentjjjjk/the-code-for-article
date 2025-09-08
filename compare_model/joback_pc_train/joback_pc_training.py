import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
from thermo import Joback
from collections import defaultdict

warnings.filterwarnings('ignore')

np.random.seed(42)

column = 'pc'
file_path = f'matched_compounds_data_{column}.csv'

data = pd.read_csv(file_path)

# Extract Joback group features from SMILES
def extract_joback_features(smiles_list):
    all_groups = set()
    group_counts_list = []
    atom_counts_list = []
    
    # First collect all possible group types
    for smiles in smiles_list:
        try:
            J = Joback(smiles)
            all_groups.update(J.counts.keys())
        except:
            pass
    
    all_groups = sorted(list(all_groups))
    
    # Extract group counts and atom counts for each molecule
    for smiles in smiles_list:
        try:
            J = Joback(smiles)
            group_counts = [J.counts.get(group, 0) for group in all_groups]
            atom_count = J.atom_count
            group_counts_list.append(group_counts)
            atom_counts_list.append(atom_count)
        except:
            # If parsing fails, fill with zeros
            group_counts_list.append([0] * len(all_groups))
            atom_counts_list.append(0)
    
    return np.array(group_counts_list), np.array(atom_counts_list), all_groups

# Extract features
group_features, atom_counts, group_names = extract_joback_features(data['SMILES'])

# Build feature matrix: group counts + atom counts
X = np.column_stack([group_features, atom_counts])
y = data[column].values

print(f"Discovered group types: {group_names}")
print(f"Number of groups: {len(group_names)}")

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

# Custom Joback model class
class JobackModel:
    def __init__(self):
        self.group_weights = None
        self.atom_weight = None
        self.bias = None
        self.group_names = None
    
    def fit(self, X, y):
        # The last column of X is atom count, previous columns are group counts
        # According to Joback formula: p_c = (∑w_i N_i + w N_atoms + b)^-2
        # We need to fit: sqrt(1/p_c) = ∑w_i N_i + w N_atoms + b
        
        y_transformed = np.sqrt(1.0 / y)  # Transform target variable
        
        # Use linear regression to fit transformed target
        lr = LinearRegression()
        lr.fit(X, y_transformed)
        
        self.group_weights = lr.coef_[:-1]  # Group weights
        self.atom_weight = lr.coef_[-1]     # Atom weight
        self.bias = lr.intercept_           # Bias
        
        return self
    
    def predict(self, X):
        # Calculate linear combination
        linear_combination = (np.dot(X[:, :-1], self.group_weights) + 
                            X[:, -1] * self.atom_weight + 
                            self.bias)
        
        # Apply Joback formula: p_c = (linear_combination)^-2
        # Avoid division by zero
        linear_combination = np.maximum(linear_combination, 1e-10)
        return 1.0 / (linear_combination ** 2)

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
    
    # Use custom Joback model
    model = JobackModel()
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

# Output best model parameters
if best_model is not None:
    print("\nBest model parameters:")
    print(f"Atom weight: {best_model.atom_weight:.6f}")
    print(f"Bias: {best_model.bias:.6f}")
    print("Group weights:")
    for i, group_name in enumerate(group_names):
        print(f"  Group {group_name}: {best_model.group_weights[i]:.6f}")
