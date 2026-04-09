import pandas as pd
import pathlib

# Load the volume file
volume_file = "/Users/anishr/EV Charging Demand /UrbanEVDataset/20220901-20230228_zone-cleaned-aggregated/charge_1hour/volume.csv"

print("Loading dataset...")
df = pd.read_csv(volume_file)

# We have time and many zone columns. Let's aggregate all zones into a single 'demand' column.
print("Aggregating demand...")
df['demand'] = df.drop(columns=['time']).sum(axis=1)
df = df[['time', 'demand']].rename(columns={'time': 'timestamp'})

# Save to ev_data.csv
output_path = "/Users/anishr/EV Charging Demand /ev-demand-project/ev_data.csv"
df.to_csv(output_path, index=False)
print(f"Saved aggregated data to {output_path}")
