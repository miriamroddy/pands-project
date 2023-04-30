import pandas as pd

# Load the iris dataset from CSV file
Irisdata = pd.read_csv("iris.csv")

# Compute summary statistics for each variable
summary_stats = Irisdata.describe()

# Open a text file for writing the summary statistics
with open("summary_stats.txt", "w") as f:
    # Loop through each variable and write its summary statistics to the text file
    for column in summary_stats.columns:
        f.write(f"Variable: {column}\n")
        f.write(f"{summary_stats[column]}\n\n")

# Print a message to confirm that the summary statistics have been exported to the text file
print("Summary statistics have been exported to 'summary_stats.txt'")
