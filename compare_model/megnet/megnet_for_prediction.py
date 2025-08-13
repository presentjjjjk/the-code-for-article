import tensorflow as tf
import pandas as pd
import numpy as np
from rdkit import RDLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from convert_csv_to_json import column_name
RDLogger.DisableLog('rdApp.*')

column_name = 'Vc'

# Load data
data = pd.read_json(f'molecules_{column_name}.json')

from openbabel import pybel
structures = [pybel.readstring('xyz', x) for x in data['xyz']]
targets = (data[column_name]*1000).tolist() 

import sys
sys.path.append(r'c:\Users\34331\Desktop\CES_return\compare_model\megnet\megnet')

from megnet.data.molecule import MolecularGraph
from megnet.models import MEGNetModel

# Initialize cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

# Track best model
best_r2 = -float('inf')
best_model = None

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(kf.split(structures)):
    print(f"\nFold {fold + 1}/10")
    
    # Split data according to cross-validation indices
    train_structures = [structures[i] for i in train_index]
    test_structures_temp = [structures[i] for i in test_index]
    train_targets = [targets[i] for i in train_index]
    test_targets_temp = [targets[i] for i in test_index]
    
    # Further split test data into dev and test (50:50) - matching MLP(BP).py
    dev_structures, test_structures, dev_targets, test_targets = train_test_split(
        test_structures_temp, test_targets_temp, test_size=0.5, random_state=fold
    )
    
    # Create model for this fold
    model = MEGNetModel(27, 2, 30, nblocks=3, lr=1e-2,
                        n1=16, n2=8, n3=4, npass=3, ntarget=1,
                        graph_converter=MolecularGraph())
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-9
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    # Train the model with custom learning rate scheduler
    model.train(
        train_structures, 
        train_targets, 
        validation_structures=dev_structures,
        validation_targets=dev_targets,
        epochs=1000, 
        verbose=1,
        batch_size=32,
        save_checkpoint=False,
        callbacks=[early_stopping, lr_scheduler]

    )
    
    # Predict on test set
    batch_size = len(test_structures)
    y_pred = model.predict_structures(test_structures, batch_size=batch_size, pbar=True)
    y_pred = np.array(y_pred).flatten()
    y_test = np.array(test_targets)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"Fold {fold + 1} - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Store metrics
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

print(f"\nBest model: R² = {best_r2:.4f})")

# Print final results
print(f"Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")


