import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
csv_file_path = 'day_data.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Plot the line graph without the "Time (seconds)" column
plt.plot(df['Time Multiplied by 3 (seconds)'], df['Density (%)'], marker='o', linestyle='-', color='b')

# Set plot labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Density (%)')
plt.title('Density over Day Time')

# Show the plot
plt.grid(True)
plt.show()
