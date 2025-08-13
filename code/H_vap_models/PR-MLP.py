from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout,Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import max_error
# import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Set default data types to 64-bit
tf.keras.backend.set_floatx('float64')
np.random.seed(42)

# Set random seed
def set_random_seed(seed_value=42):
    """
    Set all relevant random seeds to ensure reproducibility of results
    Args:
        seed_value: Random seed value
    """
    # 1. Set Python random seed
    random.seed(seed_value)

    # 2. Set numpy random seed
    np.random.seed(seed_value)

    # 3. Set tensorflow random seed
    tf.random.set_seed(seed_value)

    # 4. Set Python environment variables
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Settings for GPU
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 5. Set tensorflow configuration
    tf_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    tf.compat.v1.set_random_seed(seed_value)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=tf_config)
    tf.compat.v1.keras.backend.set_session(sess)

# Call function to set random seed
set_random_seed(42)

file_path = 'dataset_H_vap.csv'
data = pd.read_csv(file_path,encoding='latin1')

# Features
X = data[[
    # Molecular polarity descriptors
    'Dipole Moment (Debye)',
    'Polarizability',
    'free_energy',
    # Topological descriptors
    'Chi0v',
    'Chi0n',
    'Chi1v',
    'Chi1n',
    'Chi2v',
    'Chi2n',
    'Chi3v',
    'Chi3n',
    'Chi4v',
    'Chi4n',
    'Kappa1','Kappa2',
    'HallKierAlpha',
    'BalabanJ',
    # Molecular surface area descriptors
    'TPSA',
    'LabuteASA',
    'SMR_VSA1', # Surface area with high negative charge
    'SMR_VSA9', # Surface area with high positive charge
    'PEOE_VSA1',
    'PEOE_VSA14',
    # Hydrogen bond descriptors
    'HBD','HBA',
    # Structural information descriptors
    'Num Rotatable Bonds',
    'Num Aromatic Atoms','Num Aromatic Rings','Num Aromatic Bonds',
    'BertzCT',
    'Volume',
    'Sphericity',
    # Electrostatic potential descriptors
    "Min_value",
    "Max_value",
    "Overall_surface_area",
    "Positive_surface_area",
    "Negative_surface_area",
    "Overall_average",
    "Positive_average",
    "Negative_average",
    "Overall_variance",
    "Positive_variance",
    "Negative_variance",
    "Balance",
    "Internal_charge_separation",
    "MPI",
    "Nonpolar_surface_area",
    "Polar_surface_area",
    "Overall_skewness",
    "Positive_skewness",
    "Negative_skewness",
    # Additional descriptors - LogP, Molecular Weight
    'LogP',
    'Molecular Weight',
]]

y=data['H_vap']*1000

import pickle

# Load pretrained model and Scaler
pretrained_model = load_model('best_model.h5')
with open('best_scaler.pkl', 'rb') as f:
    pretrained_scaler = pickle.load(f)

# Feature scaling
X_total_scaled = pretrained_scaler.transform(X)

# Predict boiling point
predicted_boiling_point = pretrained_model.predict(X_total_scaled)

# Ensure prediction values are a one-dimensional array
if predicted_boiling_point.ndim > 1:
    predicted_boiling_point = predicted_boiling_point.flatten()

# Add predicted values to feature matrix
X['predicted_boiling_point'] = predicted_boiling_point
# Load pretrained model and Scaler
pretrained_model = load_model('best_model_Tc.h5')
with open('best_scaler_Tc.pkl', 'rb') as f:
    pretrained_scaler = pickle.load(f)

# Feature scaling
X_total_scaled = pretrained_scaler.transform(X)

# Predict critical temperature
predicted_Tc = pretrained_model.predict(X_total_scaled)

# Ensure prediction values are a one-dimensional array
if predicted_Tc.ndim > 1:
    predicted_Tc = predicted_Tc.flatten()

# Add predicted values to feature matrix
X['Tc'] = predicted_Tc

# Load pretrained model and Scaler
pretrained_model = load_model('best_model_pc.h5')
with open('best_scaler_pc.pkl', 'rb') as f:
    pretrained_scaler = pickle.load(f)

# Feature scaling
X_total_scaled = pretrained_scaler.transform(X.iloc[:, [i for i in range(X.shape[1]) if i != X.shape[1] - 2]])  # Exclude second-to-last column, keep last column

# Predict critical pressure
predicted_pc = pretrained_model.predict(X_total_scaled)

# Ensure prediction values are a one-dimensional array
if predicted_pc.ndim > 1:
    predicted_pc = predicted_pc.flatten()

# Add predicted values to feature matrix
X['pc'] = predicted_pc

batch_size = 512


# Initialize cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

best_r2=-float('inf')


# Function to add Gaussian noise
def add_gaussian_noise(X, noise_factor=0.02):
    """
    Add Gaussian noise to input data
    Args:
        X: Input data
        noise_factor: Noise intensity factor
    Returns:
        Data with added noise
    """
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


# Calculate enthalpy of vaporization using PR equation
@tf.function
def solve_cubic_equation(a, b, R, T, p_param):
    # Core coefficient calculation remains unchanged
    c2 = b - (R * T) / p_param
    c2_div3 = c2 / tf.constant(3.0, dtype=tf.float64)  # Pre-compute frequently used value
    
    # Use more efficient cubic calculation (multiplication instead of pow)
    c1 = a / p_param - tf.constant(3.0, dtype=tf.float64) * tf.square(b) - tf.constant(2.0, dtype=tf.float64) * b * R * T / p_param
    b_cubed = b * b * b  # Optimize cubic calculation
    c0 = b_cubed + (R * T * tf.square(b)) / p_param - (a * b) / p_param

    # Reconstructed p,q calculation (mathematically equivalent but numerically more stable)
    p = c1 - tf.constant(3.0, dtype=tf.float64) * tf.square(c2_div3)  # Eliminate large number subtraction
    q_c2_div3_cubed = c2_div3 * c2_div3 * c2_div3  # Optimize cubic calculation
    q = tf.constant(2.0, dtype=tf.float64) * q_c2_div3_cubed - c1 * c2_div3 + c0

    # Enhanced numerical stability calculation
    sqrt_input = tf.maximum(-p / tf.constant(3.0, dtype=tf.float64), tf.constant(1e-12, dtype=tf.float64))  # Ensure non-negative input
    sqrt_neg_p = tf.sqrt(sqrt_input)
    
    # Optimized denominator calculation (add small constant to prevent division by zero)
    sqrt_neg_p_cubed = sqrt_neg_p * sqrt_neg_p * sqrt_neg_p  # Multiplication instead of pow
    denominator = tf.constant(2.0, dtype=tf.float64) * sqrt_neg_p_cubed + tf.constant(1e-12, dtype=tf.float64)
    
    # Limit the range of the inverse trigonometric function input
    argument = tf.clip_by_value(-q / denominator, tf.constant(-1.0+1e-8, dtype=tf.float64), tf.constant(1.0-1e-8, dtype=tf.float64))  # Stricter boundary protection
    theta = tf.acos(argument)

    # Vectorized angle calculation
    k_vals = tf.constant([0.0, 1.0, 2.0], dtype=tf.float64)
    angles = (tf.expand_dims(theta, -1) + tf.constant(2.0, dtype=tf.float64) * tf.constant(np.pi, dtype=tf.float64) * k_vals) / tf.constant(3.0, dtype=tf.float64)
    cos_terms = tf.cos(angles)
    
    # Final root calculation (keep broadcast mode)
    roots = tf.constant(2.0, dtype=tf.float64) * tf.expand_dims(sqrt_neg_p, -1) * cos_terms - tf.expand_dims(c2_div3, -1)
    
    return tf.sort(roots, axis=1)

@tf.function
def process_roots_batch(roots, b_batch):
    epsilon = tf.constant(1e-8, dtype=tf.float64)
    batch_size = tf.shape(roots)[0]

    # Alternative to ensure denominator is not zero
    safety_ratio = tf.constant(1.001, dtype=tf.float64)
    valid_b = tf.maximum(b_batch, tf.constant(1e-6, dtype=tf.float64))  # Ensure b is not zero or negative

    # Convert to complex and extract real part
    roots_real = tf.math.real(roots)
    roots_imag = tf.math.imag(roots)
    
    # Mark valid real roots (imaginary part absolute value<1e-6 considered real root)
    real_roots_mask = tf.abs(roots_imag) < tf.constant(1e-6, dtype=tf.float64)
    # Mark strictly greater than b roots (consider numerical stability)
    valid_roots_mask = (roots_real > tf.expand_dims(valid_b, 1) + epsilon) & real_roots_mask
    
    # Validity conditions
    all_real = tf.reduce_all(real_roots_mask, axis=1)
    all_valid = tf.reduce_all(valid_roots_mask, axis=1)
    valid_condition = tf.logical_and(all_real, all_valid)
    
    # Valid case handling
    masked_roots = tf.where(valid_roots_mask, roots_real, tf.constant(tf.float64.max, dtype=tf.float64))
    sorted_roots = tf.sort(masked_roots, axis=1)
    V_sl_valid = sorted_roots[:, 0]
    V_sv_valid = sorted_roots[:, 2]

    # Invalid case handling (use physical constraint values)
    V_sl_invalid = valid_b * safety_ratio + epsilon
    V_sv_invalid = valid_b * tf.constant(1000.0, dtype=tf.float64)  # Sufficiently large but still maintains physical meaning
    
    V_sl = tf.where(valid_condition, V_sl_valid, V_sl_invalid)
    V_sv = tf.where(valid_condition, V_sv_valid, V_sv_invalid)

    # Final validity verification
    final_valid = (V_sl > valid_b) & (V_sv > valid_b) & (V_sl < V_sv)
    V_sl = tf.where(final_valid, V_sl, V_sl_invalid)
    V_sv = tf.where(final_valid, V_sv, V_sv_invalid)

    return V_sl, V_sv

@tf.function
def calc_H_tf(input_tensor, factor):
    Tb = input_tensor[:, 0]
    Tc = input_tensor[:, 1]
    pc = input_tensor[:, 2] / tf.constant(10.0, dtype=tf.float64)  # bar to MPa

    R = tf.constant(8.31451, dtype=tf.float64)
    a0 = tf.constant(0.457235, dtype=tf.float64) * (R * Tc)**2 / pc
    b = tf.constant(0.077796, dtype=tf.float64) * R * Tc / pc
    k = tf.constant(0.378893, dtype=tf.float64) + tf.constant(1.4897153, dtype=tf.float64) * factor - tf.constant(0.17131848, dtype=tf.float64) * tf.square(factor) + tf.constant(0.0196554, dtype=tf.float64) * tf.pow(factor, tf.constant(3.0, dtype=tf.float64))

    term_inside = tf.constant(1.0, dtype=tf.float64) + k * (tf.constant(1.0, dtype=tf.float64) - tf.sqrt(Tb / Tc))
    term_inside = tf.maximum(term_inside, tf.constant(1e-12, dtype=tf.float64))  # Prevent negative numbers
    alpha = tf.square(term_inside)
    a = alpha * a0

    d_alpha_dT = -k * term_inside / tf.sqrt(Tb * Tc)
    da_dT = a0 * d_alpha_dT

    # Initial estimate to ensure p_s is positive
    p_s_initial = pc**(tf.constant(10.0,dtype=tf.float64)/tf.constant(3.0,dtype=tf.float64)-tf.constant(7.0,dtype=tf.float64)/tf.constant(3.0,dtype=tf.float64)*Tc/Tb)* tf.constant(10.0, dtype=tf.float64) ** (tf.constant(7.0, dtype=tf.float64) * (tf.constant(1.0, dtype=tf.float64) + factor) / tf.constant(3.0, dtype=tf.float64) * (tf.constant(1.0, dtype=tf.float64) - Tc / Tb))
    p_s_initial = tf.maximum(p_s_initial, tf.constant(1e-6, dtype=tf.float64))  # Ensure initial value is not negative

    # Iteration setup
    max_iterations = 20
    converged = tf.zeros_like(p_s_initial, dtype=tf.bool)
    iteration = tf.constant(0)

    def cond(p_s, converged, iteration):
        return tf.logical_and(
            tf.less(iteration, max_iterations),
            tf.reduce_any(tf.logical_not(converged))
        )

    def body(p_s, converged, iteration):
        roots = solve_cubic_equation(a, b, R, Tb, p_s)

        V_sl,V_sv = process_roots_batch(roots,b)

        Z_sv = p_s * V_sv / (R * Tb)
        Z_sl = p_s * V_sl / (R * Tb)

        sqrt2 = tf.sqrt(tf.constant(2.0, dtype=tf.float64))
        log_term_sv = tf.math.log(tf.maximum((p_s * (V_sv - b)) / (R * Tb), tf.constant(1e-12, dtype=tf.float64)))
        log_ratio_sv = tf.math.log(tf.maximum((V_sv + (sqrt2 + tf.constant(1.0, dtype=tf.float64)) * b) / (V_sv + (tf.constant(1.0, dtype=tf.float64) - sqrt2) * b), tf.constant(1e-12, dtype=tf.float64)))
        ln_phi_sv = Z_sv - tf.constant(1.0, dtype=tf.float64) - log_term_sv - a / (tf.constant(2.0, dtype=tf.float64)**tf.constant(1.5, dtype=tf.float64) * b * R * Tb) * log_ratio_sv

        log_term_sl = tf.math.log(tf.maximum((p_s * (V_sl - b)) / (R * Tb), tf.constant(1e-12, dtype=tf.float64)))
        log_ratio_sl = tf.math.log(tf.maximum((V_sl + (sqrt2 + tf.constant(1.0, dtype=tf.float64)) * b) / (V_sl + (tf.constant(1.0, dtype=tf.float64) - sqrt2) * b), tf.constant(1e-12, dtype=tf.float64)))
        ln_phi_sl = Z_sl - tf.constant(1.0, dtype=tf.float64) - log_term_sl - a / (tf.constant(2.0, dtype=tf.float64)**tf.constant(1.5, dtype=tf.float64) * b * R * Tb) * log_ratio_sl

        ln_phi_diff = ln_phi_sv - ln_phi_sl
        abs_diff = tf.abs(ln_phi_diff)
        newly_converged = abs_diff < tf.constant(1e-6, dtype=tf.float64)
        new_converged = tf.logical_or(converged, newly_converged)

        Z_diff = Z_sv - Z_sl + tf.constant(1e-12, dtype=tf.float64)
        C = tf.constant(1.0, dtype=tf.float64) - (ln_phi_diff) / Z_diff
        C_clipped = tf.clip_by_value(C, tf.constant(0.5, dtype=tf.float64), tf.constant(1.5, dtype=tf.float64))
        active_mask = tf.logical_not(new_converged)
        new_p_s = tf.where(active_mask, p_s * C_clipped, p_s)
        new_p_s = tf.maximum(new_p_s, tf.constant(1e-6, dtype=tf.float64))  # Prevent p_s from becoming negative or zero

        return new_p_s, new_converged, iteration + 1

    p_s_final, converged, _ = tf.while_loop(
        cond, body,
        loop_vars=(p_s_initial, converged, iteration),
        maximum_iterations=max_iterations
    )

    # Final calculation
    roots_final = solve_cubic_equation(a, b, R, Tb, p_s_final)
    V_sl_final,V_sv_final = process_roots_batch(roots_final,b)

    Z_liq = p_s_final * V_sl_final / (R * Tb)
    Z_vap = p_s_final * V_sv_final / (R * Tb)

    sqrt2 = tf.sqrt(tf.constant(2.0, dtype=tf.float64))
    log_term_liq = tf.math.log(tf.maximum((V_sl_final + (sqrt2 + tf.constant(1.0, dtype=tf.float64)) * b) / (V_sl_final + (tf.constant(1.0, dtype=tf.float64) - sqrt2) * b), tf.constant(1e-12, dtype=tf.float64)))
    H_res_liq = R * Tb * (Z_liq - tf.constant(1.0, dtype=tf.float64)) - (a - Tb * da_dT) / (tf.constant(2.0, dtype=tf.float64) * sqrt2 * b) * log_term_liq

    log_term_vap = tf.math.log(tf.maximum((V_sv_final + (sqrt2 + tf.constant(1.0, dtype=tf.float64)) * b) / (V_sv_final + (tf.constant(1.0, dtype=tf.float64) - sqrt2) * b), tf.constant(1e-12, dtype=tf.float64)))
    H_res_vap = R * Tb * (Z_vap - tf.constant(1.0, dtype=tf.float64)) - (a - Tb * da_dT) / (tf.constant(2.0, dtype=tf.float64) * sqrt2 * b) * log_term_vap
    delta_H = (H_res_vap - H_res_liq) 
    delta_H = tf.where(
    tf.math.is_finite(delta_H), 
    delta_H, 
    tf.zeros_like(delta_H)  # Replace NaN with 0 or a reasonable default value
    )
    return delta_H

fold_histories = []  # Used to save the training history of each fold

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_temp = X.iloc[train_index,:-3], X.iloc[test_index,:-3] # Take the front part first

    # Then normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_temp_scaled = scaler.transform(X_temp)
 
    # Then do feature concatenation
    X_train_scaled = np.hstack([X_train_scaled, X.iloc[train_index,-3:].values])
    X_temp_scaled = np.hstack([X_temp_scaled, X.iloc[test_index,-3:].values])

    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

    X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(
        X_temp_scaled, y_temp, test_index, test_size=0.5, random_state=fold
    )

    # Data augmentation: Add noisy samples
    X_train_noisy = add_gaussian_noise(X_train_scaled)
    X_train_augmented = np.vstack([X_train_scaled, X_train_noisy])
    y_train_augmented = np.concatenate([y_train, y_train])

    # Build the neural network model
    factor_model = Sequential([
        Dense(128, input_dim=X.shape[1]-3, 
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            kernel_initializer='he_normal', dtype='float64'),  # Keep intermediate layer normal initialization
        
        BatchNormalization(dtype='float64'),
        Activation('relu', dtype='float64'),
        Dropout(0.3),
        
        Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.05),
            kernel_initializer='he_normal', dtype='float64'),
        BatchNormalization(dtype='float64'),
        Activation('relu', dtype='float64'),
        Dropout(0.2),
        
        Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer='he_normal', dtype='float64'),
        BatchNormalization(dtype='float64'),
        Activation('relu', dtype='float64'),
        Dropout(0.1),
        
        Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer='he_normal', dtype='float64'),
        BatchNormalization(dtype='float64'),
        Activation('relu', dtype='float64'),
        
        # Key modification: Initialize the last layer to 0
        Dense(1, 
            kernel_initializer='zeros',  # Initialize weight matrix to 0
            bias_initializer='zeros',    # Initialize bias to 0
            activity_regularizer=tf.keras.regularizers.l2(0.5), dtype='float64'),  # Add output regularization
        
        # Add smooth tanh constraint to ensure output is in the range [-1, +∞)
        tf.keras.layers.Lambda(lambda x: tf.where(x < 0, tf.tanh(x), x), dtype='float64')
    ])

    # Construct the final model
    # Model architecture adjustment
    inputs = Input(shape=(X.shape[1],), dtype='float64')
    input_features = inputs[:, -3:]  # (None, 3)
    factor_features = inputs[:, :-3]

    # Add dimension verification layer
    factor = factor_model(factor_features)
    factor = tf.keras.layers.Reshape([1])(factor)
    factor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(factor)

    # Final output layer
    output = tf.keras.layers.Lambda(
        lambda x: calc_H_tf(x[0], x[1]),
    )([input_features, factor])

    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, 
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        global_clipnorm=0.5  # Add global gradient clipping
    )

    # Add learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-9
    )

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Set up EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    # Train the model with EarlyStopping on the development set
    history = model.fit(
        X_train_augmented, y_train_augmented,  # Use augmented data
        epochs=2000,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_dev, y_dev),
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate the model on the test fold
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Convert to percentage

    # Store the metrics for this fold
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)

    fold_histories.append([history,r2])

    # Save the best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_scaler = scaler
 
print(f"Best model saved:R² = {best_r2:.4f})")

# Calculate and display the mean and standard deviation of R² and MAE
print(f"Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")

for i in range(len(fold_histories)):
    if fold_histories[i][1]==best_r2:
        best_history = fold_histories[i][0]
        break

import matplotlib.pyplot as plt

# Draw the loss curve of the best model
plt.figure(figsize=(8, 6))

# Get training and validation loss
train_loss = best_history.history['loss']
val_loss = best_history.history['val_loss']

# Use log scale
plt.yscale('log')

# Draw the curve
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Development Loss')

# Set the range of the Y-axis to avoid extreme values affecting
plt.ylim(bottom=min(train_loss + val_loss) * 0.9, top=max(train_loss + val_loss) * 1.1)

# Add labels, title and legend
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE, log scale)')
plt.title('Best Model - Loss Curve (Log Scale)')
plt.legend()
plt.grid(True)

# Save the image and display
plt.savefig('Best_Model_Loss_Curve(H_vap).png', dpi=300, bbox_inches='tight')
plt.show()

# Load the entire dataset using the best scaler
X_0 = X.iloc[:,:-3]
X_0_scale = best_scaler.transform(X_0)
X_tot_scaled = np.hstack([X_0_scale, X.iloc[:,-3:].values])

# Use the best model to make predictions on the entire dataset
H_pre = best_model.predict(X_tot_scaled)

# Extract the predicted values
final_predictions = H_pre.flatten()
final_actuals = y.values.flatten()

# Calculate absolute errors
abs_errors = np.abs(final_predictions - final_actuals)

# Visualize the comparison between predicted and actual values
plt.figure(figsize=(8, 6))

# Set chart style
plt.style.use('default')  # Switch to default style

# Create gradient scatter plot
scatter = plt.scatter(final_actuals, final_predictions, 
                     alpha=0.8,                          
                     s=30,                              
                     c=abs_errors,                   # Change to absolute errors
                     cmap='ocean',                    
                     edgecolor='white',                 
                     linewidth=0.5)                     

# Add ideal prediction line
min_val = min(final_actuals.min(), final_predictions.min())
max_val = max(final_actuals.max(), final_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 
         color='#FF6B6B',                              
         linestyle='-',                               
         linewidth=1.5,                                
         label="ideal line")

# Optimize labels and title
plt.xlabel('Actual H_vap (J/mol)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted H_vap (J/mol)', fontsize=12, fontweight='bold')
plt.title('Neural Networks Model Prediction vs Actual H_vap(Entire Dataset)', fontsize=14, pad=15)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Absolute Error (J/mol)', fontsize=10)  # Changed label

# Optimize grid lines
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust margins
plt.tight_layout()
# Add the legend and save the plot
plt.legend(loc='upper left')
plt.savefig('Actual_vs_Predicted_H_vap_Enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

