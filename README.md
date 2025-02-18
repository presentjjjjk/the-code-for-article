# Manual

This repository stores the supporting code for the paper titled 'Highly generalizable model with cascaded feature engineering: toward small datasets.

The deep learning framework we use is tensorflow_gpu 2.10.0 under Windows system,and we used Nvidia RTX4070 graphics card for training.If you want to run the above code on your machine, please note that the results may vary slightly depending on the hardware.

The code mainly contains three models, namely boiling point prediction, critical parameter prediction, and evaporation enthalpy prediction. The article mentions that different models under the same category can be obtained by changing the annotations.

The dataset can be found in the supporting information of the paper

We have added several comments in the code of the critical parameter training model. You can select some of the comments to get the model in our paper,for example, you can comment out this section in MLP(T_c).py: 

```python
import pickle
# Load pre-trained model and Scaler
pretrained_model = load_model('best_model.h5')
with open('best_scaler.pkl', 'rb') as f:
    pretrained_scaler = pickle.load(f)

# Feature scaling
X_total_scaled = pretrained_scaler.transform(X)

# Predict boiling point
predicted_boiling_point = pretrained_model.predict(X_total_scaled)

# Ensure predictions are 1-dimensional array
if predicted_boiling_point.ndim > 1:
    predicted_boiling_point = predicted_boiling_point.flatten()

# Add predictions to feature matrix
X['predicted_boiling_point'] = predicted_boiling_point
```
This way you can get the results of a direct training run without boiling point feature.Here we provide the running results of MLP(T_c).py for your reference:

```
Best model saved: R² = 0.9819)
Average R² Score: 0.9453 ± 0.0361
Average MAE: 14.4485 ± 3.9488
Average RMSE: 20.0864 ± 6.4914
Average Max Error: 49.9595 ± 19.5027
Average Mean Relative Error: 2.63% ± 0.65%
```

All the model calculation results and generated images have been presented in the paper. In addition, we have also uploaded the saved model weights and normalizers to the repository.


The following three pictures visualize part of the model framework

![Boiling point-critical temperature joint training framework](https://github.com/user-attachments/assets/186109a7-d907-452c-8e49-4e5c370b52ab)

![Critical temperature and critical pressure joint training framework](https://github.com/user-attachments/assets/220debfe-d94b-43f4-ad66-23bf8a085020)

![PR-MLP (1)](https://github.com/user-attachments/assets/ecce384f-6ffc-44e1-ac8a-90a606d312df)



