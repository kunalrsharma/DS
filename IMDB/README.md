# IMDB project
The aim here is to predict movie ratings from the data at IMDB. 

## Assumptions and data cleaning:
1) The files consdiered for analysis are: title.ratings.tsv, title.crew.tsv and title.basics.tsv.
2) After merging the files, the data set is cleaned of '\N'.
3) Then the following columns are dropped: 'endYear','originalTitle', iisAdult' and 'writers'. The latter is 
   dropped much later into the project after analysis.
4) 'primaryTitle' is replaced by its length
5) 'directors' and 'genres' are replaced by frequency of each key.They alternatively could be grouped too but the distribution
    suggested us to go with frequency count. 
6) Study of correlation heat map indicates insignificant role of 'primaryTitle' so it is dropped. 
7) Eventually, the feature space consists of following variables: 
   'tconst','numVotes','directors','startYear','runtimeMinutes','genres'. 
8) Finally, for medeling a regression approach is taken. Alternatively, a classification approach could be taken. 

