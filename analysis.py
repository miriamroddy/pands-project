
###################################################################
## Task: Write a program called analysis.py that:                 #
## 1. Outputs a summary of each variable to a single text file,   #
## 2. Saves a histogram of each variable to png files, and        #
## 3. Outputs a scatter plot of each pair of variables.           #
## 4. Performs any other analysis you think is appropriate        # 
###################################################################

## We start by importing all the necessary modules - see README for discussion.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, normaltest, f_oneway, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from statsmodels.formula.api import ols


## We load data from a CSV file named 'iris.csv' into a pandas DataFrame called 'Irisdata':

Irisdata = pd.read_csv('iris.csv')

## Open the file (descriptive_stats.txt) in write mode using the open() function and the with statement. This ensures that the file is properly closed after the block of code 
## is executed, even if an exception occurs.

with open('descriptive_stats.txt', 'w') as f:

## We then want to write the structure of the dataset to our file (to be read by colleagues, collaborators etc.). Irisdata.info(buf=f) is called so we can get the structure 
## of the dataset Irisdata. The info() method provides information about the dataset, including the number of rows, columns, column names, data types, and memory usage.

    f.write('Dataset Structure:\n\n')
    Irisdata.info(buf=f)
    f.write('\n\n')


## We need to check if there are missing values in the DF. If hasMissingValues is True, we know that the DF contains at least one missing value. 
## If it's False, it means that there are no missing values.

    hasMissingValues = Irisdata.isnull().values.any()

## We then use an if statement to write information about the presence or absence of missing values to the file. We use a header to indicate we're in the mssing values section, 
## then write either "Yes" or "No" (depending on whether we have missing values or not). We add a newline character (\n) to keep things tidy.

    f.write('Missing Values:\n')
    if hasMissingValues:
        f.write('Yes\n')
    else:
        f.write('No\n')
    f.write('\n')

## We now want to see if we have any duplicated observations in our dataset. The code Irisdata.duplicated() checks each row in the DF to see if there are any duplicates of any previous row. 
    
    duplicateCount = Irisdata.duplicated().sum()

## We then want to highlight that the following info relates to duplicate rows. We do this by specifying the count of duplicate rows found in the dataset, so we know how much of a problem
## this might cause.

    f.write('Duplicates:\n')
    f.write(str(duplicateCount))
    f.write(' duplicates found.\n\n')

## We can now indicate the exact rows which are duplicated. We write those duplicated rows as a string to our file. 

    f.write('Duplicated Lines:\n')
    f.write(str(Irisdata[Irisdata.duplicated()]))
    f.write('\n\n')

## We use Irisdata.describe() to compute various summary statistics of our variables, including count, 
## mean, standard deviation, minimum, quartiles, and maximum. We also specify that we want these rounded to two decimal places.

    f.write('Descriptive Statistics:\n\n')
    f.write(str(Irisdata.describe().round(2)))
    f.write('\n\n')

## We want to have the right pandas options for formatting the descriptive , so we specify: 
## (1) the display width (1000 characters per row) (2) that we want all columns displayed

    pd.set_option('display.width', 1000)  
    pd.set_option('display.max_columns', None)   

## Next, we want to get information about the descriptive statistics by species. The groupby('species') groups the data by species 
## and describe() calculates descriptive stats for each group. 

    f.write('Descriptive Statistics by Species:\n\n')
    f.write(str(Irisdata.groupby('species').describe().round(2)))
    f.write('\n\n')

## We are finished with the display width and display max columns settings so we reset them to the defaults:

    pd.reset_option('display.width')
    pd.reset_option('display.max_columns')


## We now want to calculate z-scores for each value in the DF, so we can get a sense of how far values are from their means. We make a new DF called zScoresdata using the results of
## this next line:

zScores = stats.zscore(Irisdata.iloc[:, :-1])
zScoresdata = pd.DataFrame(zScores, columns=Irisdata.columns[:-1]) ## the new DF is called zScoresdata

## Now that we have zscores, we want to highlight anything that might be an outlier. If any z-score is above three (see README for discussion of the chosen threshold)
## we consider it to be a possible outlier. With "(zScoresdata.abs() > 3)", we are comparing each absolute z-score value in our DF to the threshold of 3. 
## possible_outliers is a subset of our Irisdata DF that contains any instance of a z-score greater than 3. 

possible_outliers = Irisdata[(zScoresdata.abs() > 3).any(axis=1)]

## Add details of our calculations to descriptive_stats.txt - we open back desciptive_stats.txt in append mode (a)

with open('descriptive_stats.txt', 'a') as f:
    f.write('\n')
    f.write('Outliers using Z-scores:\n')

## We check if possible_outliers is empty by using an if statement - IF there are, we convert the DF to a string and write it tp descriptive_stats.txt

    if not possible_outliers.empty:
        f.write(str(possible_outliers))

## Otherwise we write a message to the file indicating that we didn't find any outliers in the dataset

    else:
        f.write('No outliers found in the dataset based on z-scores.')
    f.write('\n\n')

## Now we will include a couple of tests of normality, starting with the D'Agostino-Pearson test

    f.write("D'Agostino-Pearson Test:\n\n")

    ## We now want to use the stats.normaltest() function from the scipy.stats - this contains the D'Agostino-Pearson normality test.
    dAgostinoResults = Irisdata.iloc[:, :-1].apply(stats.normaltest) ## We the :-1 portion to select all columns except the last one 
    ## (We want to exclude the column that contains the species name).

    f.write(str(dAgostinoResults.round(2))) ## I like to round to two decimal places to be consistent
    f.write('\n\n')

    ## We now want to go through the process again, but this time divided by species. See README for theoretical underpinnings of this.

    f.write("D'Agostino-Pearson Test by Species:\n\n")

    ## We are using a lambda function to perform the D'Agostino-Pearson test for each species. The iloc indexer is used to select specific rows and columns within function
    ## to apply to each group of data (i.e. species)

    dAgostinoResultsSpecies = Irisdata.groupby('species').apply(lambda x: x.iloc[:, :-1].apply(stats.normaltest).round(2))
    f.write(str(dAgostinoResultsSpecies))
    f.write('\n\n')

    ## We now want to perform our second text of normality - the Shapiro-Wilk test.

    f.write("Shapiro-Wilk Test:\n\n")

    ## We call on stats.shapiro from the scipy.stats module to perform this test
    shapiroResults = Irisdata.iloc[:, :-1].apply(stats.shapiro) ## Again, we are excluding the last column of our dataset
    f.write(str(shapiroResults.round(2)))
    f.write('\n\n')

    ## Now we want to perform the test divided by species.

    f.write("Shapiro-Wilk Test by Species:\n\n")
    shapiroResultsSpecies = Irisdata.groupby('species').apply(lambda x: x.iloc[:, :-1].apply(stats.shapiro).round(2))
    f.write(str(shapiroResultsSpecies))
    f.write('\n\n')

## ANOVA - List of variables to analyze
variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

## Open the file in append mode
file_name = "descriptive_stats.txt"
with open(file_name, "a") as f:

## we start with a for loop to Perform ANOVA for each variable

    for variable in variables:
        f.write(f"Variable: {variable}\n")
        f.write("------------------------------------\n")

## Perform separate ANOVA for each species - we use ols from statsmodel.api

        for species in Irisdata['species'].unique():
            subset_data = Irisdata[Irisdata['species'] == species]
            model = ols(f'{variable} ~ 1', data=subset_data).fit()
            anova_result = sm.stats.anova_lm(model)

            f.write(f"Species: {species}\n")
            f.write(str(anova_result))
            f.write("\n")

        f.write("====================================\n")




## This is the end of us writing to our descriptive_stats file. Because we used with open, we know that the file will ve automaticall closed so we don't 
## need to explicitly request this.

##################################################################################################
#                                    Vizualisations                                              #
##################################################################################################

## Now we will create various visualizations. 
##############
#  Boxplots  #
##############

## To assuage boredom and to attempt for some consistency, we want to create a colour palette with nice colours - 
## I'm choosing pinks/purples. We are telling seaborn specifically to use this palette:

colourpalette = ["#E687A5", "#CC5C8F", "#8B5A9B", "#5E3C78"]
sns.set_palette(colourpalette)


# We start with the creation of a boxplot for each variable by species. We start with sepal length:
# We tell seaborn that we want species on the x-axis and sepal length on the y.

sns.boxplot(x="species", y="sepal_length", data=Irisdata)
plt.title("Sepal Length by Species") # We create the title for the plot here
plt.ylabel("Sepal Length") # The label for the y-axis
plt.savefig("Boxplot_Sepal_Length.png") # We save the plot to a PNG from matplotlib.pyplot
plt.show() ## This shows the currently active figure. This wasn't requested specifically but I find it handy to make sure all the code is working without checking the pngs.

## We repeat for sepal width.

sns.boxplot(x="species", y="sepal_width", data=Irisdata)
plt.title("Sepal Width by Species")
plt.ylabel("Sepal Width")
plt.savefig("Boxplot_Sepal_Width.png")
plt.show()

## And for petal length

sns.boxplot(x="species", y="petal_length", data=Irisdata)
plt.title("Petal Length by Species")
plt.ylabel("Petal Length")
plt.savefig("Boxplot_Petal_Length.png")
plt.show()

## And petal width

sns.boxplot(x="species", y="petal_width", data=Irisdata)
plt.title("Petal Width by Species")
plt.ylabel("Petal Width")
plt.savefig("Boxplot_Petal_Width.png")
plt.show()

##################
#   Histograms   #
##################

#  Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Reference my colour palette again
colourpalette = ["#E687A5", "#CC5C8F", "#8B5A9B", "#5E3C78"]  # We'll go with the palette I already used
## Four colours are defined here but it will default to the first one a (pale pink) .

# Set the plot style. We'll use the whitegrid style to add grid lines with a white background. 

sns.set_style("whitegrid")

# We make a list which will contain each of our variables.
variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

## We use a for loop so we can iterate over each variable in our list.

for variable in variables:
    plt.figure(figsize=(8, 6))

## We use Seaborn's histplot function to create a histogram for each variable in our list, using 
## the first colour in our palette [the 0 indicates that it's the first colour].

    sns.histplot(data=iris, x=variable, color=colourpalette[0], kde=True)

## We amend the variable names so they are in title case without an underscore:

    formatted_var = variable.replace('_', ' ').title()

## Specify the title and labels of our histograms.

    plt.title(f"Histogram of {formatted_var}") ## The title of the histogram, which will change during 
## each iteration of the loop
    plt.xlabel(formatted_var) ## We have the amended variable on the x-axis
    plt.ylabel("Frequency") ## We will have frequency of occurance on the y axis
    plt.savefig(f"histogram_{variable}.png") ## We save the file to a .PNG
    plt.show() ## Again, this was handy during development to make sure that the code worked

####################################################
#              Overlapping histograms              #
####################################################

# Now we create a figure with subplots for each variable 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8)) ## we want 2 rows and 2 column, with dimensions of 8*10 inches
fig.suptitle("Overlapping Histograms by Species") # Set the title for the figure

# Create overlapping histograms for each variable separated by species. We separates the different shades/hues by species -it referes to the "species" column in our 
## DF. The x-axis has the variable names. The step element specifies that we want to use a step function. We specify that we want a KDE plot so that we can create a smooth curve
# that follows the shape of the ditro to see the extent to which the data approximates a normal distribution. The ax=axs figures tell seaborn where to place each plot.

sns.histplot(data=Irisdata, x="sepal_length", hue="species", element="step", kde=True, ax=axs[0,0])
sns.histplot(data=Irisdata, x="sepal_width", hue="species", element="step", kde=True, ax=axs[0,1])
sns.histplot(data=Irisdata, x="petal_length", hue="species", element="step", kde=True, ax=axs[1,0])
sns.histplot(data=Irisdata, x="petal_width", hue="species", element="step", kde=True, ax=axs[1,1])

# Then we define the labels for both axes
axs[0,0].set(xlabel='Sepal Length (cm)', ylabel='Frequency')
axs[0,1].set(xlabel='Sepal Width (cm)', ylabel='Frequency')
axs[1,0].set(xlabel='Petal Length (cm)', ylabel='Frequency')
axs[1,1].set(xlabel='Petal Width (cm)', ylabel='Frequency')

plt.savefig("overlappinghistograms.png") # Save as a PNG and show plots
plt.show()

# Set a custom color palette with pink and purple colors
## my_palette = ['#E687A5', '#CC5C8F', '#8B5A9B']
## sns.set_palette(my_palette)

#################################
#          Scatterplots         #
#################################

# We wants to look at scatterplots for each pair of variables now. We'll start with a scatterplot of petal length vs petal width, with each species getting a different hue:

sns.scatterplot(data=Irisdata, x='petal_length', y='petal_width', hue='species')

## Add a title and labels
plt.title('Scatterplot of Petal Length and Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# We want the current legend handles and labels in title case so we retrieve the default legend and handles first:
handles, labels = plt.gca().get_legend_handles_labels()

# Change the first letter of the legend labels to uppercase
labels = [label.capitalize() for label in labels]

# Define the updated legend labels
plt.legend(handles, labels)

# Save the scatterplot as a PNG file
plt.savefig('Scatterplot_Petal.png') ## Save as a .PNG
plt.show() 

# Now we want to same again but this time, we have a scatterplot of sepal length vs sepal width
sns.scatterplot(data=Irisdata, x='sepal_length', y='sepal_width', hue='species')

# Set the title and labels for the plot
plt.title('Scatterplot of Sepal Length and Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# We want the current legend handles and labels in title case so we retrieve the default legend and handles first:
handles, labels = plt.gca().get_legend_handles_labels()

# Change the first letter of the legend labels to uppercase
labels = [label.capitalize() for label in labels]

# Define the updated legend labels
plt.legend(handles, labels)

# Save the scatterplot as a PNG file
plt.savefig('Scatterplot_Sepal.png')
plt.show()

#################################################################
#                     Pairplot                                 #
#################################################################


# Create a pairplot of the iris dataset, with separate scatter plots for each species
plot = sns.pairplot(data=Irisdata, hue='species', diag_kind='hist')

# Save the plot as a PNG file
plot.savefig('pairplot.png')

# Display the plot
plt.show()

#######################################
#            Heatmap                  #
#######################################

# We need to explude the 'species' column from our DF because the data is not numerical
iris = iris.drop('species', axis=1)

# We now calculate the correlation matrix of the remaining columns.
corr = iris.corr()

# Create a heatmap - this time we'll use an inbuilt palette (RdPu) from Seaborn to switch things up
cmap = sns.color_palette("RdPu", as_cmap=True)
## We use seaborn's heatmap function to visualise the correlation matrix using a colour coded grid.
sns.heatmap(corr, annot=True, cmap=cmap) # the annot=True adds numerical annotations

# Save the figure and display it
plt.savefig('heatmap.png')
plt.show()


####################################
#          Machine Learning        #
####################################

# We start by Separating the features (X) and the target variable (y)
X = Irisdata.drop('species', axis=1)
y = Irisdata['species']

## we split our dataset into two sets - the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##  we create a KNN classifier with k=3- ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
knn = KNeighborsClassifier(n_neighbors=3)

## Fit the model to our training data
knn.fit(X_train, y_train)

## Make predictions on our test data
y_pred = knn.predict(X_test)

## Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

## We print the score to the console
print("Accuracy:", accuracy)
