import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("cdclfullbody.csv")

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

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)
X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
y_tensor_test = torch.tensor(y_test, dtype=torch.float32)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

pinn = PINN()
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = pinn(X_tensor)

    data_loss = torch.mean((y_pred - y_tensor) ** 2)

    penalty_Cd_negative = torch.relu(-y_pred[:,0])
    penalty_Cd_upper = torch.relu(y_pred[:,0] - 2.0)
    penalty_Cl_upper = torch.relu(y_pred[:,1] - 4.0)
    penalty_Cl_lower = torch.relu(-4.0 - y_pred[:,1])

    physics_loss = torch.mean(
        penalty_Cd_negative**2 +
        penalty_Cd_upper**2 +
        penalty_Cl_upper**2 +
        penalty_Cl_lower**2
    )

    total_loss = data_loss + 1.0 * physics_loss
    train_losses.append(total_loss.item())

    total_loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Data Loss {data_loss.item():.6f}, Physics Loss {physics_loss.item():.6f}")

with torch.no_grad():
    final_preds = pinn(X_tensor_test)
    print("\nPredicted Cd and Cl:\n", final_preds.numpy())
    print("\nTrue Cd and Cl:\n", y_tensor.numpy())

import sklearn.metrics as metrics

with torch.no_grad():
    preds_np = final_preds.numpy()
    true_np = y_tensor_test.numpy()

    mse = metrics.mean_squared_error(true_np, preds_np)
    print(f"\nMean Squared Error (MSE): {mse:.6f}")

    mae = metrics.mean_absolute_error(true_np, preds_np)
    print(f"Mean Absolute Error (MAE): {mae:.6f}")

    r2 = metrics.r2_score(true_np, preds_np)
    print(f"R^2 Score: {r2:.6f}")

    print("\nPer-output metrics:")
    for i, name in enumerate(["Cd", "Cl"]):
        mse_i = metrics.mean_squared_error(true_np[:, i], preds_np[:, i])
        mae_i = metrics.mean_absolute_error(true_np[:, i], preds_np[:, i])
        r2_i = metrics.r2_score(true_np[:, i], preds_np[:, i])
        print(f"  {name}: MSE={mse_i:.6f}, MAE={mae_i:.6f}, R^2={r2_i:.6f}")

import matplotlib.pyplot as plt
import torch
import numpy as np
print(y_tensor_test)
print(X_tensor_test)

y_actual_np = y_tensor_test.detach().cpu().numpy()
y_predicted_np = final_preds.detach().cpu().numpy()

num_samples = y_actual_np.shape[0]
sample_indices = np.arange(num_samples)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(sample_indices, y_actual_np[:, 0], label='Actual Cd', color='blue', linewidth=2)
plt.plot(sample_indices, y_predicted_np[:, 0], label='Predicted Cd', color='orange', linestyle='--', linewidth=1.5)
plt.xlabel("Sample Index")
plt.ylabel("Cd Value")
plt.title("Actual vs Predicted Cd")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sample_indices, y_actual_np[:, 1], label='Actual Cl', color='green', linewidth=2)
plt.plot(sample_indices, y_predicted_np[:, 1], label='Predicted Cl', color='red', linestyle='--', linewidth=1.5)
plt.xlabel("Sample Index")
plt.ylabel("Cl Value")
plt.title("Actual vs Predicted Cl")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

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
