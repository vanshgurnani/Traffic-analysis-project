import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV files
data1 = pd.read_csv("day_accuracy.csv")
data2 = pd.read_csv("night_accuracy.csv")

# Multiply accuracy values by 100
data1['Accuracy'] *= 100
data2['Accuracy'] *= 100

# Extracting density and accuracy data
density1 = data1['Density']
accuracy1 = data1['Accuracy']
density2 = data2['Density']
accuracy2 = data2['Accuracy']

# Plotting the graph for Day Data
plt.figure(figsize=(8, 6))
plt.plot(accuracy1, density1, label='Day Data', marker='o', linestyle='-')
plt.title('Density vs. Accuracy (Day Data)')
plt.xlabel('Accuracy (%)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('day_accuracy_graph.png')
plt.show()

# Plotting the graph for Night Data
plt.figure(figsize=(8, 6))
plt.plot(accuracy2, density2, label='Night Data', marker='x', linestyle='-')
plt.title('Density vs. Accuracy (Night Data)')
plt.xlabel('Accuracy (%)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('night_accuracy_graph.png')
plt.show()
