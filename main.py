import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

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



# Finalize plot
ax.grid(True)
plt.legend()
plt.show()
