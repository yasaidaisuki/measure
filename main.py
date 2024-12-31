import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Initialize the plot
fig, ax = plt.subplots()
fig.set_size_inches(12, 5)

# Figure labels
ax.set_title("Frequency Response")
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('dBr')

# Axis scale & ticks
ax.yaxis.set_major_locator(ticker.MaxNLocator())
plt.xscale('log')
plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], 
           [20, 50, 100, 200, 500, '1k', '2k', '5k', '10k', '20k'])
plt.xlim(20, 20000)

# Read data
df = pd.read_csv('./targets/Diffuse field 5128.csv')
df_raw = pd.read_csv('./measurements/Focal Elex.csv')

# Ensure data alignment
if not df['frequency'].equals(df_raw['frequency']):
    raise ValueError("Frequency columns in the datasets do not match!")

df['equalized'] = df['raw'] - df_raw['raw']

######################## OBJECTIVE ANALYSIS ########################
'''
deviation/variation calculations using numpy, pandas, matplotlib
'''

# General RMSE and MAE
rmse = np.sqrt(np.mean(df['equalized']**2))
mae = np.mean(abs(df['equalized']))

# Frequency band analysis
bands = {
    "Sub-Bass": (20, 60),
    "Bass": (60, 250),
    "Lower Mids": (250, 500),
    "Mids": (500, 2000),
    "Upper Mids": (2000, 5000),
    "Treble": (5000, 10000),
    "High Treble": (10000, 20000),
}

targets = {
    "Sub-Bass": 3,
    "Bass": 3,
    "Lower Mids": 0,
    "Mids": 0,
    "Upper Mids": 0,
    "Treble": -2,
    "High Treble": -5,
}

grades = {}

# Helper function
def get_deviation(target, raw):
    return abs(target - raw)

for band, (lb, ub) in bands.items():
    # Filter data within the frequency range
    band_data = df[(df["frequency"] >= lb) & (df["frequency"] <= ub)]
    
    if band_data.empty:
        print(f"No data for band: {band}")
        continue

    # Calculate the average dB value in the band
    avg_dB = band_data["equalized"].mean()
    
    # Get target dB and deviation
    target_dB = targets[band]
    deviation = get_deviation(target_dB, avg_dB)
    
    # Assign grade
    if deviation < 1:
        grade = 5
    elif deviation < 3:
        grade = 4
    elif deviation < 5:
        grade = 3
    else:
        grade = 2
    
    # Store results
    grades[band] = {"Avg dB": avg_dB, "Target dB": target_dB, "Deviation": deviation, "Grade": grade}

mean_grade = 0

# Print results
for band, result in grades.items():
    mean_grade = result['Grade'] + mean_grade 
    print(f"{band}: {result}")

mean_grade = mean_grade/len(bands)

print(mean_grade)

# Plot based on user input
read = input("Display Diffused or Raw: ").strip().lower()
if read == "diffused":
    plt.plot(df['frequency'], df['equalized'], label='Equalized')
elif read == "raw":
    plt.plot(df_raw['frequency'], df_raw['raw'], label='Raw')
else:
    print("Invalid input. Showing raw data.")
    plt.plot(df_raw['frequency'], df_raw['raw'], label='Raw')


######################## tonal balance and preferences analysis using pytorch ######################## 

''' POC 
# Example Neural Network
class HeadphoneSoundModel(nn.Module):
    def __init__(self):
        super(HeadphoneSoundModel, self).__init__()
        self.fc1 = nn.Linear(100, 64)  # Adjust input size to match data shape
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # For regression, output a single value (e.g., score)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prepare your data (frequency response -> ratings/labels)
# Example frequency response data (100 frequencies)
x_train = torch.rand((100, 100))  # 100 samples, 100 frequency values per sample
y_train = torch.rand(100, 1)  # Ratings or scores for each headphone

# Create DataLoader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# Model, loss, and optimizer
model = HeadphoneSoundModel()
loss_fn = nn.MSELoss()  # Use CrossEntropyLoss if classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluate model on test data
'''

# Finalize plot
ax.grid(True)
plt.legend()
plt.show()
