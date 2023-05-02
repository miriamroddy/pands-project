import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro, normaltest, norm

# Load the iris dataset into a Pandas DataFrame
Irisdata = pd.read_csv('iris.csv')

# Check for missing values
if Irisdata.isnull().values.any():
    missing_msg = 'Uh-oh. This dataset contains missing values.'
else:
    missing_msg = 'The dataset does not contain missing values.'

# Check for data entry errors
if (Irisdata.describe().loc[['min', 'max']] < 0).values.any():
    error_msg = 'Uh-oh. This dataset contains data entry errors.\n'
else:
    error_msg = 'The dataset does not contain data entry errors.\n'

# Compute summary statistics and round them to two decimal places
descriptive_stats = Irisdata.describe().round(2)

# We add a new line after the variable headings
descriptive_stats_str = descriptive_stats.to_string(header='\n\n', index=False)

# Write the missing value and data entry error messages and summary statistics to a text file
with open('descriptive_stats.txt', 'w') as f:
    print(missing_msg, file=f)
    print('', file=f)
    print(error_msg, file=f)
    print(descriptive_stats_str, file=f)


# Extract values of sepal length column
sepal_length = Irisdata['sepal_length'].values

# We  test for normality with use the Shapiro-Wilks test
shapiro_test = shapiro(sepal_length)
print('Shapiro-Wilk test statistic:', shapiro_test.statistic)
print('Shapiro-Wilk test p-value:', shapiro_test.pvalue)

# D'Agostino-Pearson normality test
dago_test = normaltest(sepal_length)
print('D\'Agostino-Pearson test statistic:', dago_test.statistic)
print('D\'Agostino-Pearson test p-value:', dago_test.pvalue)


# List of variable names
variable_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


## Load the Iris dataset from Seaborn
Irisdata = sns.load_dataset('iris')

# Loop through each column in the dataset and create a histogram with normal distribution curve
for col in Irisdata.columns[:-1]:
    fig, ax = plt.subplots()
    
    # Plot the histogram of the variable
    sns.histplot(data=Irisdata, x=col, kde=True, ax=ax)
    
    # Create the normal distribution curve
    mu, std = norm.fit(Irisdata[col])
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    # Plot the normal distribution (bell curve)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title(f'{col} Histogram with Normal Distribution')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    
    plt.show()

