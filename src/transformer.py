import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt

# Set device for GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')
df_product = pd.read_csv('/content/product.csv')
df_geography = pd.read_csv('/content/geography.csv')

# Combine and aggregate by date
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.groupby('Date').agg(
    {'Units': 'sum', 'Revenue': 'sum', 'COGS': 'sum'}).reset_index()
full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.sort_values('Date', inplace=True)
full_df.set_index('Date', inplace=True)

# Feature engineering: add day of week, month encoding, lagged revenue, and holidays
days_of_week = np.array([d.weekday()
                        # 0-6 (Mon-Sun)
                         for d in full_df.index]).reshape(-1, 1)
month_sin = np.sin(2 * np.pi * full_df.index.month / 12).values.reshape(-1, 1)
month_cos = np.cos(2 * np.pi * full_df.index.month / 12).values.reshape(-1, 1)
lag1_revenue = full_df['Revenue'].shift(1).fillna(
    full_df['Revenue'].mean()).values.reshape(-1, 1)
lag7_revenue = full_df['Revenue'].shift(7).fillna(
    full_df['Revenue'].mean()).values.reshape(-1, 1)
holidays = [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-07-04'),
            pd.to_datetime('2022-01-01'), pd.to_datetime('2022-07-04')]
is_holiday = full_df.index.isin(holidays).astype(int).reshape(-1, 1)

# Combine all features
features = np.hstack((full_df[['Units', 'COGS']], days_of_week, month_sin, month_cos,
                     lag1_revenue, lag7_revenue, is_holiday))
target = full_df['Revenue'].values

# Log transform target to handle outliers
target_log = np.log1p(target)

# Scale features and target
features_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features_scaled = features_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target_log.reshape(-1, 1))
dataset = np.hstack((features_scaled, target_scaled))

# Split into sequences for input (30 days) and output (1 day)


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-
                                 1], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


n_steps_in, n_steps_out = 30, 1
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# 8 features (Units, COGS, day, month_sin, month_cos, lag1, lag7, holiday)
n_features = X.shape[2]

# Split into train, validation, and test sets (last 535 days for test)
test_size = 535
train_size = len(X) - test_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create DataLoader for batch training
train_dataset = TensorDataset(torch.FloatTensor(X_train_main).to(device),
                              torch.FloatTensor(y_train_main).to(device))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(torch.FloatTensor(X_test).to(device),
                             torch.FloatTensor(y_test).to(device))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Transformer model for revenue prediction


class RevenueTransformer(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, sequence_length):
        super(RevenueTransformer, self).__init__()
        # Linear layer to embed features into higher dimension for Transformer
        self.embedding = nn.Linear(n_features, d_model)
        # Generate position encoding for sequence order
        self.position_enc = self._get_position_encoding(
            sequence_length, d_model).to(device)
        # Define Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=0.1, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)
        # Final layer to predict revenue
        self.predictor = nn.Linear(d_model, 1)

    def _get_position_encoding(self, sequence_length, d_model):
        # Generate position encoding using sine and cosine functions
        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(sequence_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # Transpose input for Transformer (sequence_length, batch_size, n_features)
        x = x.transpose(1, 0)
        # Embed features into higher dimension
        embedded_x = self.embedding(x)
        # Add position encoding (broadcast across batch)
        embedded_x += self.position_enc.unsqueeze(1)
        # Process through Transformer encoder
        output = self.transformer_encoder(embedded_x)
        # Take last time step's output for prediction
        last_output = output[-1, :, :]
        # Predict revenue
        prediction = self.predictor(last_output)
        return prediction


# Initialize model with hyperparameters
d_model = 64  # Dimension of the model
nhead = 4     # Number of attention heads
num_layers = 2  # Number of encoder layers
sequence_length = n_steps_in
model = RevenueTransformer(n_features, d_model, nhead,
                           num_layers, sequence_length).to(device)

# Define RMSE loss for training


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


criterion = RMSELoss()
optimizer = ADOPT(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20)

# Training loop with early stopping
num_epochs = 1500
best_test_loss = float('inf')
best_epoch = 0
patience = 50
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_y).item()
    test_loss /= len(test_loader)

    scheduler.step(test_loss)

    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}')

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_model.pt',
                   _use_new_zipfile_serialization=False)
        patience_counter = 0
    # else:
    #     patience_counter += 1
    #     if patience_counter >= patience:
    #         print(f"Early stopping at epoch {epoch+1}")
    #         break

# Load best model with security setting
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
print(f"Best model saved from epoch: {best_epoch}")

# Test the model
model.eval()
with torch.no_grad():
    y_pred_scaled = model(torch.FloatTensor(X_test).to(device))
    y_pred = np.expm1(target_scaler.inverse_transform(
        y_pred_scaled.cpu().numpy()))  # Reverse log transform
    y_test_unscaled = np.expm1(target_scaler.inverse_transform(y_test))

y_test_flat = y_test_unscaled.flatten()
y_pred_flat = y_pred.flatten()

# Compute evaluation metrics
r2 = r2_score(y_test_flat, y_pred_flat)
rmse = sqrt(mean_squared_error(y_test_flat, y_pred_flat))
mape = mean_absolute_percentage_error(y_test_flat, y_pred_flat)
print(f'R²: {r2:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}')

# Plot actual vs predicted revenue
test_df_grouped = test_df.groupby('Date').agg(
    {'Units': 'sum', 'Revenue': 'sum', 'COGS': 'sum'}).reset_index()
test_dates = pd.to_datetime(test_df_grouped['Date']).values
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_flat, label='Actual Revenue')
plt.plot(test_dates, y_pred_flat, label='Predicted Revenue', linestyle='-')
plt.title('Transformer Model - Actual vs Predicted Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
