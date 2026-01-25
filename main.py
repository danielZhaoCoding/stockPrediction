import math
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import seaborn as sns
import torch
import yfinance as yf
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})

# Plot line charts
df_plot = df.copy()

ncols = 2
nrows = int(round(df_plot.shape[1] / ncols, 0))

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
for i, ax in enumerate(fig.axes):
        sns.lineplot(data = df_plot.iloc[:, i], ax=ax)
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.tight_layout()
plt.show()

# Indexing Batches
train_df = df.sort_values(by=['Date']).copy()

# List of considered Features
FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume']

print('FEATURE LIST')
print([f for f in FEATURES])

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]

# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['Close']

#print the tail of the dataframe
data_filtered_ext.tail()

# Get the number of rows in the data
nrows = data_filtered.shape[0]

# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(data_filtered_ext['Close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

# Set the sequence length - this is the timeframe used to make a single prediction
sequence_length = 50

# Prediction Index
index_Close = data.columns.get_loc("Close")

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data
train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

# Create the training and test data
train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len - sequence_length:, :]

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction

    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)

# Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Validate that the prediction value and the input match up
# The last close price of the second input sample should equal the first prediction value
print(x_train[1][sequence_length-1][index_Close])
print(y_train[0])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # initialize hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # initialize num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # initialize fully connected layer 1, input is hidden size output is 5
        self.fc1 = nn.Linear(hidden_size, 5)
        self.fc2 = nn.Linear(5, output_size)
        # initialize fully connected layer 2

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Preprocessing steps (assuming data_filtered, and the train and test splits are done)
scaler = MinMaxScaler()
np_data_unscaled = np.array(data_filtered)
np_data_scaled = scaler.fit_transform(np_data_unscaled)

x_train, y_train = partition_dataset(sequence_length, train_data)  # Assuming partition_dataset is defined
x_test, y_test = partition_dataset(sequence_length, test_data)

n_neurons = x_train.shape[1] * x_train.shape[2]
input_size = x_train.shape[2]
hidden_size = n_neurons // 2
num_layers = 2
output_size = 1
learning_rate = 1e-4

# instantiate the lstm model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
# create the loss function with L1Loss
criterion = nn.L1Loss()
# create the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Convert to Torch tensors
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epochs = 25

for epoch in range(epochs):
    for x_batch, y_batch in train_dataloader:

        # compute model predictions
        # reset the gradients of the model parameters to zero before backpropagation
        # compute the gradient of the loss with respect to the model parameters enabling backpropagation
        # updates the model parameters

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Get the predicted values
y_pred_scaled = model(x_test)
y_pred_scaled_numpy = y_pred_scaled.detach().numpy()

# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled_numpy)
y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))



MAE = mean_absolute_error(y_test_unscaled, y_pred)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')


# The date from which on the date is displayed
display_start_date = "2019-01-01"

# Add the difference between the valid and predicted prices
train = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
valid = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
valid.insert(1, "y_pred", y_pred, True)
valid.insert(1, "y_test", y_test_unscaled, True)
valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
df_union = pd.concat([train, valid])

# Zoom in to a closer timeframe
df_union_zoom = df_union[df_union.index > display_start_date]

# Create the lineplot
fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("y_pred vs y_test")
plt.ylabel(stockname, fontsize=18)
sns.set_palette(["#090364", "#1960EF", "#EF5919"])
sns.lineplot(data=df_union_zoom[['y_pred', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)

# Create the bar plot with the differences
df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom["residuals"].dropna()]
ax1.bar(height=df_union_zoom['residuals'].dropna(), x=df_union_zoom['residuals'].dropna().index, width=3, label='residuals', color=df_sub)
plt.legend()
plt.show()


df_temp = df[-sequence_length:]
new_df = df_temp[FEATURES]

N = sequence_length
last_N_days = new_df[-sequence_length:].values
# Make predictions using the model
for i in range(7):
  last_N_days_scaled = scaler.transform(last_N_days)

  # Create an empty list and Append past N days
  X_test_new = []
  X_test_new.append(last_N_days_scaled)

  # Convert the X_test data set to a numpy array and reshape the data
  X_test_new = torch.from_numpy(np.array(X_test_new)).float()


  pred_price_scaled = model(X_test_new)
  pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.detach().numpy().reshape(-1, 1))
  pred_unscaled_scalar = float(pred_price_unscaled.squeeze())

  # Ensure last_N_days is a tensor
  if isinstance(last_N_days, np.ndarray):
      last_N_days = torch.from_numpy(last_N_days.astype(np.float32))

  # Extract scalar values
  pred_unscaled_scalar = float(pred_price_unscaled.squeeze())
  pred_scaled_scalar = float(pred_price_scaled.squeeze())
  if i == 0:
    print(f'One Day Prediction: {pred_price_unscaled[0][0]}')
  last_feature_3 = float(last_N_days[-1][3])  # already converted above

  # Create new row tensor
  new_row = torch.tensor([
      pred_unscaled_scalar + 20000,
      pred_unscaled_scalar - 20000,
      last_feature_3,
      pred_scaled_scalar,
      200.0
  ]).unsqueeze(0)  # shape: (1, 5)

  # Concatenate
  last_N_days = torch.cat([last_N_days, new_row], dim=0)


# Print last price and predicted price for the next day
price_today = np.round(new_df['Close'].iloc[-1], 2)
predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
change_percent = np.round(100 - (price_today * 100)/predicted_price, 2).iloc[0]

plus = '+'; minus = ''
print(f'The close price for {stockname} at {end_date} was {price_today.iloc[0]}')
print(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')


