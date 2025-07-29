#full body---------------------------------------------------------------------


#!pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# 1) Load CSV data
df = pd.read_csv("cdclfullbody.csv")

# Extract input features (forces and moments)
input_columns = [
    "PRESSURE_FORCE_X", "PRESSURE_FORCE_Y", "PRESSURE_FORCE_Z",
    "VISCOUS_FORCE_X", "VISCOUS_FORCE_Y", "VISCOUS_FORCE_Z",
    "PRESSURE_MOMENT_X", "PRESSURE_MOMENT_Y", "PRESSURE_MOMENT_Z",
    "VISCOUS_MOMENT_X", "VISCOUS_MOMENT_Y", "VISCOUS_MOMENT_Z"
]
X = df[input_columns].values
y = df[["FORCE_COEFFICIENT_CD", "FORCE_COEFFICIENT_CL"]].values

train_losses = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_tensor = torch.tensor(X_train, dtype=torch.float32)# tensors utlize gpu which makes it faster
y_tensor = torch.tensor(y_train, dtype=torch.float32)# tensors can be convered to any dimension. similar to array
X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
y_tensor_test = torch.tensor(y_test, dtype=torch.float32)
# 2) Create PINN neural network
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)  # output Cd, Cl
        )
    def forward(self, x):
        return self.net(x)

pinn = PINN()
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)



epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = pinn(X_tensor)  # [N, 2]: Cd, Cl

    # Data loss: fit true Cd, Cl directly
    data_loss = torch.mean((y_pred - y_tensor) ** 2)

    # Existing physics constraint: penalize negative Cd
    penalty_Cd_negative = torch.relu(-y_pred[:,0])  # negative Cd → positive penalty

    # New domain constraint: Cd above 2 → penalty
    penalty_Cd_upper = torch.relu(y_pred[:,0] - 2.0)

    # New domain constraint: Cl outside [-4, +4] → penalty
    penalty_Cl_upper = torch.relu(y_pred[:,1] - 4.0)
    penalty_Cl_lower = torch.relu(-4.0 - y_pred[:,1])

    # Combine all penalties into physics loss
    physics_loss = torch.mean(
        penalty_Cd_negative**2 +
        penalty_Cd_upper**2 +
        penalty_Cl_upper**2 +
        penalty_Cl_lower**2
    )

    # Total loss: data + physics
    total_loss = data_loss + 1.0 * physics_loss
    train_losses.append(total_loss.item())

    total_loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Data Loss {data_loss.item():.6f}, Physics Loss {physics_loss.item():.6f}")


# 4) Print predictions
with torch.no_grad():
    final_preds = pinn(X_tensor_test)
    print("\nPredicted Cd and Cl:\n", final_preds.numpy())
    print("\nTrue Cd and Cl:\n", y_tensor.numpy())



import sklearn.metrics as metrics  # make sure scikit-learn is installed

with torch.no_grad():
    # Convert to numpy arrays
    preds_np = final_preds.numpy()
    true_np = y_tensor_test.numpy()

    # Compute MSE
    mse = metrics.mean_squared_error(true_np, preds_np)
    print(f"\nMean Squared Error (MSE): {mse:.6f}")

    # Compute MAE
    mae = metrics.mean_absolute_error(true_np, preds_np)
    print(f"Mean Absolute Error (MAE): {mae:.6f}")

    # Compute R^2
    r2 = metrics.r2_score(true_np, preds_np)
    print(f"R^2 Score: {r2:.6f}")

    # Optionally print per-output metrics if you want them separately for Cd and Cl
    print("\nPer-output metrics:")
    for i, name in enumerate(["Cd", "Cl"]):
        mse_i = metrics.mean_squared_error(true_np[:, i], preds_np[:, i])
        mae_i = metrics.mean_absolute_error(true_np[:, i], preds_np[:, i])
        r2_i = metrics.r2_score(true_np[:, i], preds_np[:, i])
        print(f"  {name}: MSE={mse_i:.6f}, MAE={mae_i:.6f}, R^2={r2_i:.6f}")


#plotting - 

import matplotlib.pyplot as plt
import torch
import numpy as np
print(y_tensor_test)
print(X_tensor_test)

y_actual_np = y_tensor_test.detach().cpu().numpy()
y_predicted_np = final_preds.detach().cpu().numpy() # Using final preds as your 'predicted' data

# Create an index for the x-axis (e.g., sample number)
num_samples = y_actual_np.shape[0]
sample_indices = np.arange(num_samples)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot(sample_indices, y_actual_np[:, 0], label='Actual Cd', color='blue', linewidth=2)
plt.plot(sample_indices, y_predicted_np[:, 0], label='Predicted Cd', color='orange', linestyle='--', linewidth=1.5)
plt.xlabel("Sample Index")
plt.ylabel("Cd Value")
plt.title("Actual vs Predicted Cd")
plt.grid(True)
plt.legend()

# --- Graph 2: Actual vs Predicted for 'Cl' (assuming it's the second column, index 1) ---
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
plt.plot(sample_indices, y_actual_np[:, 1], label='Actual Cl', color='green', linewidth=2)
plt.plot(sample_indices, y_predicted_np[:, 1], label='Predicted Cl', color='red', linestyle='--', linewidth=1.5)
plt.xlabel("Sample Index")
plt.ylabel("Cl Value")
plt.title("Actual vs Predicted Cl")
plt.grid(True)
plt.legend()

plt.tight_layout() # Adjust layout to prevent overlapping
plt.show()


import numpy as np

    # Compute RMSE
rmse = np.sqrt(mse) # RMSE is the square root of MSE
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

        # Compute MAPE (Mean Absolute Percentage Error)
    # Add a small epsilon to avoid division by zero for MAPE
epsilon = 1e-10
mape = np.mean(np.abs((true_np - preds_np) / (true_np + epsilon))) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")


import matplotlib.pyplot as plt
plt.figure(figsize=(15, 7))
plt.plot(train_losses, label='Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.show()
