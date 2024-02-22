import pandas as pd
import matplotlib.pyplot as plt

# Load day data from CSV file into a pandas DataFrame
day_csv_file_path = 'day_data.csv'  # Replace with the actual path to your day data CSV file
df_day = pd.read_csv(day_csv_file_path)

# Load night data from CSV file into a pandas DataFrame
night_csv_file_path = 'night_data.csv'  # Replace with the actual path to your night data CSV file
df_night = pd.read_csv(night_csv_file_path)

# Plot the line graph for day data
plt.plot(df_day['Time Multiplied by 3 (seconds)'], df_day['Density (%)'], color='b', label='Day')

# Plot the line graph for night data
plt.plot(df_night['Time'], df_night[' Density'], color='r', label='Night')

# Set plot labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Density (%)')
plt.title('Density over Day and Night Time')

# Show the legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
