import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files into pandas DataFrames
day_df = pd.read_csv('day_pie.csv')  # Update with your day data file name
night_df = pd.read_csv('night_pie.csv')  # Update with your night data file name

# Count occurrences of heavy vehicles in each DataFrame
day_heavy_count = (day_df['Class'].isin(['bus', 'truck'])).sum()
night_heavy_count = (night_df['Class'].isin(['bus', 'truck'])).sum()

# Plotting the pie chart
plt.figure(figsize=(10, 5))

labels = ['Day', 'Night']
sizes = [day_heavy_count, night_heavy_count]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Heavy Vehicles Distribution (Day vs Night)')

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.show()
