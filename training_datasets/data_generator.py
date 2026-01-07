import subprocess
import sys
import os

# --- PACKAGE INSTALLER SECTION ---
def install_packages():
    required_packages = ["pandas", "numpy"]
    for package in required_packages:
        try:
            # Check if package is already installed
            __import__(package)
        except ImportError:
            print(f"Package '{package}' not found. Installing now...")
            # Run the pip install command automatically
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}.")

# Run the installer
install_packages()

# Now import the libraries after they are confirmed to exist
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# --- DATA GENERATION SECTION ---
print("\nStarting dataset generation...")
start_time = time.time()

# 1. Configuration
num_rows = 1000001
# Start date from the first record in datatest.txt 
start_date = datetime(2015, 2, 2, 14, 19, 0)

# 2. Generate Timestamps with 1-minute intervals
print(f"Generating {num_rows} timestamps...")
date_list = [start_date + timedelta(minutes=i) for i in range(num_rows)]

# 3. Generate randomized data matching datatest.txt ranges [cite: 1, 2, 20]
print("Generating randomized sensor data...")
# Temperature ranges roughly 20.2 to 24.1 [cite: 1, 20]
temperature = np.random.uniform(20.2, 24.2, num_rows)
# Humidity ranges roughly 22.1 to 29.0 [cite: 1, 12]
humidity = np.random.uniform(22.1, 29.1, num_rows)
# Light is 0 at night, up to ~600 during day [cite: 1, 5]
light = np.random.uniform(0, 600, num_rows)
# CO2 ranges roughly 430 to 1176 [cite: 1, 16]
co2 = np.random.uniform(430, 1180, num_rows)
# HumidityRatio ranges roughly 0.0033 to 0.0050 [cite: 1, 16]
humidity_ratio = np.random.uniform(0.0033, 0.0051, num_rows)
# Occupancy is binary (0 or 1) [cite: 1, 5]
occupancy = np.random.randint(0, 2, num_rows)

# 4. Assemble DataFrame
df = pd.DataFrame({
    "date": date_list,
    "Temperature": temperature,
    "Humidity": humidity,
    "Light": light,
    "CO2": co2,
    "HumidityRatio": humidity_ratio,
    "Occupancy": occupancy
})

# Match original index starting at 140 
df.index = range(140, 140 + num_rows)

# 5. Export to CSV
filename = "generated_dataset_1M.csv"
print(f"Saving dataset to '{filename}'...")
df.to_csv(filename)

# 6. Final Performance Summary
end_time = time.time()
total_seconds = end_time - start_time
print(f"\nSuccess! Created {len(df)} rows.")
print(f"Total Execution Time: {total_seconds:.2f} seconds.")