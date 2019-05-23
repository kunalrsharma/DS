# IMDB project
The aim here is to predict movie ratings from the data at IMDB. 

## Assumptions and data cleaning:
1) The files consdiered for analysis are: title.ratings.tsv, title.crew.tsv and title.basics.tsv.
2) After merging the files, the data set is cleaned of duplicates, null values and '\N'.
3) Then the following columns are dropped: 'endYear','originalTitle', 'isAdult' and 'writers'. The latter is 
   dropped much later into the project after analysis.
4) 'primaryTitle' is replaced by its length
5) 'directors' and 'genres' are replaced by frequency of each key.They alternatively could be grouped too but the distribution
    suggested us to go with frequency count. 
6) Study of correlation heat map indicates insignificant role of 'primaryTitle' so it is dropped. 
7) Eventually, the feature space consists of following variables: 
   'tconst','numVotes','directors','startYear','runtimeMinutes','genres'. 
8) Finally, for modeling a regression approach is taken. Alternatively, a classification approach could be tried as well
   by treating [1,2,...,10] as class labels. (partitions of length 0.5 could be used for greater precision) 

## Analysis and Modeling:
1) First we look at the feature that's most clearly related to ratings, 'numVotes'. 
   Looking at the skewness and Kurtosis values, both of which are negative indicate
   a lef-tailed, non-normal platykurtic distribution. As a result, outlier and noise are not significant(in ratings). 
2) After simplifing the data as mentioned above, different machine learning models are tried. 
3) Firstly,  the ratings are not linear in the feature space. 
4) Secondly, it is quite safe to assume that users rating movies is a biased process. So the data contains high bias.
   Therefore, one of the best techqniques to deal with biased, non-linear data is boosting. 
5) Nevertheless, we start with OLS and Ridge rigression. 
6) OLS summary from statsmodels.api tell us that a) Jarque-Bera is too large, confirming non-normality. b) numVotes and generes play strong roles, indicated by their coef.
7) MSE for random forest train: 0.127, test: 0.945 and R^2 scores are close to 0. This is clearly overfitting. 
8) Next we try Gradient boosted trees, with staged_predict() to get the optimal tree.\
   For training -Mean squared error train: 0.9232997530025202 \
   For test- Mean squared error test: 0.9474101315226086 \
   Test error is pretty consistent with trainign error. 
9) Though R^2 is low, given the non-linearity of the data it is expected. 
10) Given an error of atmost 1 movie rating point, Gradient boosting is able to model the data pretty well. 
