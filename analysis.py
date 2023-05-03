## We start by importing all necessary libraries. See README for further details about individual libraries.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro, normaltest, norm

## We need to load our dataset into a Pandas Dataframe.

Irisdata = pd.read_csv('iris.csv')

## We check for missing values - this is good practice.

if Irisdata.isnull().values.any():
    missing_msg = 'Uh-oh. This dataset contains missing values.'
else:
    missing_msg = 'The dataset does not contain missing values.'

## We then check for data entry errors.

if (Irisdata.describe().loc[['min', 'max']] < 0).values.any():
    error_msg = 'Uh-oh. This dataset contains data entry errors.\n'
else:
    error_msg = 'The dataset does not contain data entry errors.\n'

## Compute descriptive statistics & round them to two decimal places.

descriptive_stats = Irisdata.describe().round(2)

## We wants to add a new line after the variable headings.

descriptive_stats_str = descriptive_stats.to_string(header='\n\n', index=False)

##  Use the with function to write the missing value and data entry error messages and descriptive statistics to a text file.

with open('descriptive_stats.txt', 'w') as f:
    print(missing_msg, file=f)
    print('', file=f)
    print(error_msg, file=f)
    print(descriptive_stats_str, file=f)


## Extract values from the sepal length column.

sepal_length = Irisdata['sepal_length'].values

## We test for normality with use the Shapiro-Wilks test.

shapiro_test = shapiro(sepal_length)
print('Shapiro-Wilk test statistic:', shapiro_test.statistic)
print('Shapiro-Wilk test p-value:', shapiro_test.pvalue)

## We add the D'Agostino-Pearson normality test.

dago_test = normaltest(sepal_length)
print('D\'Agostino-Pearson test statistic:', dago_test.statistic)
print('D\'Agostino-Pearson test p-value:', dago_test.pvalue)


## List of variable names.

variable_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


## Load the Iris dataset from Seaborn.

Irisdata = sns.load_dataset('iris')

## Loop through each column in the dataset and create a histogram with normal distribution curve.

for col in Irisdata.columns[:-1]:
    fig, ax = plt.subplots()
    
## Plot the histogram of all variables.

    sns.histplot(data=Irisdata, x=col, kde=True, ax=ax)
    
## We create the normal distribution curve using scipy's norm module - norm.fit applies a normal distribution to values in the Irisdata array

    mu, std = norm.fit(Irisdata[col])
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
## Plot the normal distribution (bell curve).

    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title(f'{col} Histogram with Normal Distribution')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    
    plt.show()

## Adding pairplot.

## Show the pairplots.
## We load the iris dataset - 
Iris = sns.load_dataset("iris")

## Create scatterplots for all variables.
sns.pairplot(Iris)

## Show the plots.
plt.show()