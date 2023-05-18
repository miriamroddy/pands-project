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
 
The aim of exporatory data analysis is to start to uncover patterns, relationships, and insights into the datasat, which will then help us to decide which methods to use for subsequent stages. For the Iris dataset specifically, we had the option of assuming that the data was [well balanced](https://towardsdatascience.com/eda-of-the-iris-dataset-190f6dfd946d#:~:text=The%20dataset%20is%20balanced%20i.e.,petal%20width%20and%20petal%20length), normally distributed and doesn't need much in the way of data-cleaning. It is, after all, the "Hello World" of Data Science, and wouldn't be such a commonly cited exemplar if it contained lots of messy data. However, we won't learn a whole lot from that, not is it ever good practice to merely assume - we might even find out something interesting.

I decided to combine elements of exploratory data analysis and preliminary statistical analysis in my text file, descriptive_stats.txt. I decided to examine:

Dataset Structure - The pandas [info() method](https://www.w3schools.com/python/pandas/ref_df_info.asp#:~:text=The%20info()%20method%20prints,method%20actually%20prints%20the%20info) outputs a summary of the dataset's structure to our descriptive_stats.txt file. It includes information about the numbers of rows and columns, data types and size of the dataset in KB. In a real-world scenario, this would be a useful step in documenting and sharing information with colleagues or other collaborators. If I was developing this project further, I would extract information from this basic output and format it in a way that would be more meaningful to the layman (since we were tasked with explaining the dataset to colleagues, who may not necessarily have much technical knowledge).

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
The dataset contains a total of 150 samples with an equal number of samples for each species of Iris. 

Missing Values - When conducting research, is it vital to acknowledge and address missing values so that we can be confident about the validity and reliability of our analyses. As we can see from the output, the dataset thankfully doesn't contain any missing values:

```
Missing Values:
No
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

Potential Outliers using Z-scores - A z-score is a commonly used statistical measurement which gives us information about the relative position of a data point within a dataset. A threshold of 3 is [commonly used](https://towardsdatascience.com/outlier-detection-part1-821d714524c) within data science, so I decided to go with this. We see that one set of observations is above this threshold - this gives us a little more information about what to narrow in on when we begin our data vizualisation stage.
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
Shapiro-Wilk Test - This is another test that looks for normality given a given distribution. It is based on the comparison between the observed data and the expected values under the assumption of normality. Again, because we know we are looking at different species of plants, we want to look at these figures per variable and then divided by species.
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

### Boxplots
![Boxplot - Petal Length](https://github.com/miriamroddy/pands-project/blob/main/Boxplot_Petal_Length.png)
![Boxplot - Petal Width](https://github.com/miriamroddy/pands-project/blob/main/Boxplot_Petal_Width.png)
![Boxplot - Sepal Length](https://github.com/miriamroddy/pands-project/blob/main/Boxplot_Sepal_Length.png)
![Boxplot - Sepal Width](https://github.com/miriamroddy/pands-project/blob/main/Boxplot_Sepal_Width.png)
### Histograms
![Histogram - Petal Length](https://github.com/miriamroddy/pands-project/blob/main/histogram_petal_length.png)
![Histogram - Petal Width](https://github.com/miriamroddy/pands-project/blob/main/histogram_petal_width.png)
![Histogram - Sepal Length](https://github.com/miriamroddy/pands-project/blob/main/histogram_sepal_length.png)
![Histogram - Sepal Width](https://github.com/miriamroddy/pands-project/blob/main/histogram_sepal_width.png)
### Overlapping Histogram
![Histogram - Overlapping](https://github.com/miriamroddy/pands-project/blob/main/overlappinghistograms.png)
### Scatterplots
![Scatterplot - Petal Length and Width](https://github.com/miriamroddy/pands-project/blob/main/Scatterplot_Petal.png)
![Scatterplot - Sepal Length and Width](https://github.com/miriamroddy/pands-project/blob/main/Scatterplot_Sepal.png)
### Pairplot
![Pairplot](https://github.com/miriamroddy/pands-project/blob/main/pairplot.png)
### Heatmap
![Heatmap](https://github.com/miriamroddy/pands-project/blob/main/heatmap.png)
----

## Statistical Analysis
From the table below, we see that the mean sepal length was 5.84 cm, while the mean sepal width was 3.05 cm. The mean petal length was 3.76 cm, and the mean petal width was 1.20 cm:
```
Descriptive Statistics:

       sepal_length  sepal_width  petal_length  petal_width
count        150.00       150.00        150.00       150.00
mean           5.84         3.05          3.76         1.20
std            0.83         0.43          1.76         0.76
min            4.30         2.00          1.00         0.10
25%            5.10         2.80          1.60         0.30
50%            5.80         3.00          4.35         1.30
75%            6.40         3.30          5.10         1.80
max            7.90         4.40          6.90         2.50

```
It is also important that we look at these figures divided by species - if we examine the mean and SD of the flowers as a whole, we are losing a lot of important information. Using this code: 
```
f.write(str(Irisdata.groupby('species').describe().round(2)))
```
I grouped the observations by species and we see the following output:
```
           sepal_length                                      
                  count  mean   std  min   25%  50%  75%  max
species                                                       
setosa             50.0  5.01  0.35  4.3  4.80  5.0  5.2  5.8
versicolor         50.0  5.94  0.52  4.9  5.60  5.9  6.3  7.0
virginica          50.0  6.59  0.64  4.9  6.22  6.5  6.9  7.9

```
Iris sepals (the green leaf-like structures) vary among the iris species, with setosa having the shortest sepals and virginica having the longest.
```
           sepal_width                                      
                 count  mean   std  min   25%  50%   75%  max
species                                                      
setosa            50.0  3.42  0.38  2.3  3.12  3.4  3.68  4.4
versicolor        50.0  2.77  0.31  2.0  2.52  2.8  3.00  3.4
virginica         50.0  2.97  0.32  2.2  2.80  3.0  3.18  3.8
```
Setosa generally having wider sepals compared to versicolor and virginica.
```
           petal_length                                      
                 count  mean   std  min  25%   50%   75%  max
species                                                     
setosa            50.0  1.46  0.17  1.0  1.4  1.50  1.58  1.9
versicolor        50.0  4.26  0.47  3.0  4.0  4.35  4.60  5.1
virginica         50.0  5.55  0.55  4.5  5.1  5.55  5.88  6.9
``
Setosas appear to have the shortest petals, while virginica iris flowers have the longest petals among the species.`
```
           petal_width                                      
                count  mean   std  min  25%  50%  75%  max
species                                                  
setosa           50.0  0.24  0.11  0.1  0.2  0.2  0.3  0.6
versicolor       50.0  1.33  0.20  1.0  1.2  1.3  1.5  1.8
virginica        50.0  2.03  0.27  1.4  1.8  2.0  2.3  2.5
```
When it comes to petal width, it appears that setosa iris flowers have the narrowest petals, while virginica iris flowers have relatively wider petals compared to the other species.
However, descriptive statistics don't make inferences beyond the observed sample. If we want to make inferences about the three iris species beyond the measurements that [Anderson] (https://en.wikipedia.org/wiki/Edgar_Anderson) took, we need to use statistical methods which will allow us to confidently generalise from a sample to a population. ANOVA (Analysis of Variance) is used to determine whether the means of two or more groups are significantly different from each other by examining the variation between groups compared to the variation within groups. 





- - - -

## Machine Learning Models
### KNN Classifier
Another feature of the dataset is the bimodal distribution of the petal length measurement. It is not immediately clear why the petal length measurement should be bimodal. Some researchers have hypothesized that this bimodality may be due to a combination of genetic and environmental factors, while others have suggested that it may be an artifact of the data collection process. Regardless of the cause, this feature adds an extra layer of complexity to the analysis of the iris dataset and has spurred much discussion and debate in the machine learning community.

- - - -
## Conclusion

- - - -



## References
====
- Beatty, A. (2023). Programming and Scripting [Online Higher Diploma Program]. https://vlegalwaymayo.atu.ie/course/view.php?id=6208
- Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. Annals of Eugenics, 7(2), 179-188.
- Google Python Style Guide (https://google.github.io/styleguide/pyguide.html)
- Matthes, E. (2015). Python Crash Course: A Hands-On, Project-Based Introduction to Programming. No Starch Press.
- PEP 8 -- Style Guide for Python Code - (https://www.python.org/dev/peps/pep-0008/)
- Python Style Guide by Guido van Rossum (https://legacy.python.org/dev/peps/pep-0008/#introduction)