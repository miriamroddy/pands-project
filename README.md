# Pands Project 2023



## Table of Contents

- [Introduction](#introduction)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Visualization](#data-visualization)
- [Statistical Analysis](#statistical-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Conclusion](#conclusion)


======
- - - -
## Introduction
The Iris dataset is a commonly used dataset within machine learning and pattern recognition fields. It was introduced by Ronald Fisher in 1936 as part of his research on multivariate statistics. The dataset consists of 150 observations of iris flowers, each with four measurements: sepal length, sepal width, petal length, and petal width.

The dataset is useful when exploring different machine learning algorithms, as it provides a means of comparing and evaluating the performance of different models. A task frequently cited in the literature is one that classifies the observations into one of three species of iris flowers: Iris setosa, Iris versicolor, and Iris virginica. The dataset is often used to assess the performace of different classification algorithms, including decision trees, support vector machines, and artificial neural networks.

A strength of the dataset is that it is well-balanced, with an equal number of observations for each of the three species. Since the dataset is well-balanced, it reduces the risk of bias towards one particular class. Additionally, the dataset is relatively small, which makes it easy to work with and analyse.



![Iris Species](https://github.com/miriamroddy/pands-project/blob/main/3irisimage.png)

**Libraries used**:
*Numpy* is used for performing mathematical operations and handling large arrays and matrices efficiently.  It is widely used in data analysis and machine learning applications.

The *seaborn* library focuses on data visualization. It extends the functionality of matplotlib and produces more visually appealing and informative plots. Seaborn includes functions for making many common types of plots, such as scatter plots, line plots, bar plots, histograms, and heatmaps.

*Matplotlib.pyplot* Matplotlib is a plotting library for Python. pyplot is a sub-library of Matplotlib that provides a convenient interface for creating figures, subplots, and various types of plots, such as line plots, scatter plots, bar plots, and histograms.

*Pandas* is primarily used for data manipulation and analysis - it provides data structures for efficiently storing and processing large datasets. It is built on top of NumPy and provides additional functionality for working with structured data, including tools for reading and writing data to and from various file formats, data cleaning and transformation, and data analysis.

*Scipy.stats* provides functions for statistical analysis, including the shapiro, normaltest and norm functions used in this project.

 -----------

## Exploratory Data Analysis
 
The aim of exporatory data analysis is to start to uncover patterns, relationships, and insights into the datasat, which will then help us to decide which methods to use for subsequent stages. For the Iris dataset specifically, we had the option of assuming that the data was [well balanced](https://towardsdatascience.com/eda-of-the-iris-dataset-190f6dfd946d#:~:text=The%20dataset%20is%20balanced%20i.e.,petal%20width%20and%20petal%20length), normally distributed and does't need much in the way of data-cleaning. It is after all the "Hello World" of Data Science, and wouldn't be such a commonly cited exemplar if it contained lots of messy data. However, we won't learn a whole lot from that, not is it ever good practice to merely assume - we might even find out something interesting.

I decided to combine elements of exploratory data analysis and preliminary statistical analysis in my text file, descriptive_stats.txt. I decided to examine:

Dataset Structure - The pandas info() method outputs a summary of the dataset's structure. It includes information about the numbers of rows and columns, data types and size of the dataset in KB. In a real-world scenario, this would be a useful step in documenting and sharing information with colleagues or other collaborators. If I was developing this project further, I extract information from this basic output and format it in a way that would be more meaningful to the layman (since we were tasked with explaining the dataset to colleagues).

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
```

Missing Values - When conducting research, is it vital to acknowledge and address missing values so that we can be confident about the validity and reliability of our analyses. As we can see from the output, the dataset thankfully doesn't contain any missing values:

```
Missing Values:
No
```
Data Entry Errors: 
```
Data Entry Errors: None
```
Duplicates -
```
Duplicates:
3 duplicates found.

Duplicated Lines:
     sepal_length  sepal_width  petal_length  petal_width    species
34            4.9          3.1           1.5          0.1     setosa
37            4.9          3.1           1.5          0.1     setosa
142           5.8          2.7           5.1          1.9  virginica
```
Descriptive Statistics - I will discusss this further in the Statistical Analysis section.
Potential Outliers using Z-scores -
```
        sepal_length  sepal_width  petal_length  petal_width    species
15           5.7          4.4           1.5          0.4        setosa
```
D'Agostino-Pearson Test -  This is a statistical test used to assess whether a dataset follows a normal distribution. It is based on the skewness and kurtosis of the data. It's important to conduct this analysis for each variable, and then for each variable divided by species. The reason for this is that we might find that the data is not normally distributed when assessed as a whole but the underlying reason here is that we are looking at different populations. We have a strong theoretical basis to assume that this will be the case here, since we are looking at members of three species of iris.

```
D'Agostino-Pearson Test:

   sepal_length  sepal_width  petal_length  petal_width
0          5.74         3.58        221.33       136.78
1          0.06         0.17          0.00         0.00

D'Agostino-Pearson Test by Species:

              sepal_length  sepal_width  petal_length  petal_width
species                                                           
setosa     0          0.19         1.89          2.20        13.78
           1          0.91         0.39          0.33         0.00
versicolor 0          0.84         1.45          3.32         0.33
           1          0.66         0.48          0.19         0.85
virginica  0          0.21         2.57          2.70         1.24
           1          0.90         0.28          0.26         0.54
        
```
Shapiro-Wilk Test - This is another test that looks for normality given a given distribution. It is based on the comparison between the observed data and the expected values under the assumption of normality. Againa, Bbecause we know we are looking at different spevcies of plants, we want to look at these figures per variable and again, divided by species.
```
Shapiro-Wilk Test:

   sepal_length  sepal_width  petal_length  petal_width
0          0.98         0.98          0.88          0.9
1          0.01         0.08          0.00          0.0

Shapiro-Wilk Test by Species:

              sepal_length  sepal_width  petal_length  petal_width
species                                                           
setosa     0          0.98         0.97          0.95         0.81
           1          0.46         0.20          0.05         0.00
versicolor 0          0.98         0.97          0.97         0.95
           1          0.46         0.34          0.16         0.03
virginica  0          0.97         0.97          0.96         0.96
           1          0.26         0.18          0.11         0.09
```

--------
## Data Visualization


----

## Statistical Analysis


- - - -

## Machine Learning Models
Another feature of the dataset is the bimodal distribution of the petal length measurement. It is not immediately clear why the petal length measurement should be bimodal. Some researchers have hypothesized that this bimodality may be due to a combination of genetic and environmental factors, while others have suggested that it may be an artifact of the data collection process. Regardless of the cause, this feature adds an extra layer of complexity to the analysis of the iris dataset and has spurred much discussion and debate in the machine learning community.

- - - -
## Conclusion

- - - -



## References
====

- PEP 8 -- Style Guide for Python Code - (https://www.python.org/dev/peps/pep-0008/)
- Google Python Style Guide (https://google.github.io/styleguide/pyguide.html)
- Python Style Guide by Guido van Rossum (https://legacy.python.org/dev/peps/pep-0008/#introduction)

- Matthes, E. (2015). Python Crash Course: A Hands-On, Project-Based Introduction to Programming. No Starch Press.
- Beatty, A. (2023). Programming and Scripting [Online Higher Diploma Program]. https://vlegalwaymayo.atu.ie/course/view.php?id=6208
