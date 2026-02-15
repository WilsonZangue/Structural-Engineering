import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Synthetic Data
df = pd.read_csv('daskan_master_training_set.csv')

# 2. EDA: Correlation Matrix
# This answers RQ1 by showing which features mathematically correlate with 'total_project_effort'
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix (Evidence for RQ1)')
plt.show()

# 3. EDA: Visualizing Seasonality
# Check if winter projects actually take longer (or shorter) in our data
plt.figure(figsize=(8, 5))
sns.boxplot(x='is_winter', y='total_project_effort', data=df)
plt.title('Project Effort Distribution: Summer vs. Winter')
plt.xticks([0, 1], ['Summer/Other', 'Winter'])
plt.show()

print("Phase 1 Complete: Correlation data generated.")