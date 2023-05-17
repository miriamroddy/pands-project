import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
iris = pd.read_csv("iris.csv")

# Drop the non-numeric column 'species'
iris = iris.drop('species', axis=1)

# Calculate the correlation matrix
corr = iris.corr()

# Create a heatmap using seaborn with a pink and purple color palette
cmap = sns.color_palette("RdPu", as_cmap=True)
sns.heatmap(corr, annot=True, cmap=cmap)

# Show the plot
plt.show()
