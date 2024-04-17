import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('day_pie.csv')

# Filter out the 'backpack' class
df_filtered = df[df['Class'] != 'backpack']

# Group the data by 'Class' and count the occurrences
class_counts = df_filtered['Class'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
patches, _, _ = plt.pie(class_counts, labels=None, startangle=140, autopct='', pctdistance=0.85)  # Set autopct to empty string to hide percentages inside the pie
plt.title('Day_pie_chart')

# Show legend with labels outside the pie chart
legend_labels = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(class_counts.index, class_counts / class_counts.sum() * 100)]
plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Save the image
plt.savefig('Day_pie_chart.png', bbox_inches='tight')

plt.show()
