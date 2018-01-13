## OPINION BUBBLE ENGINE 

![alt text](/bubbleviz.png)

Motivated by research on [casual information visualization](https://dl.acm.org/citation.cfm?id=1313), we built an interactive visualization of the different sides of an issue, based on people's votes on a particular claim. We combined a d3.js and vue.js front-end with regular HTTP requests to the backend to dynamically update this visualization as people voted. I built all of machine learning and math code for this, The methodology for the clustering went as follows:

* Code to process votes of form (user_id, sentence_id, vote), cluster user into groups and calculate statistics about those groups.
* Construct Binary Agree/Disagree Votes Matrix
* Perform Dimension Reduction (PCA, metric and non-metric MDS)
* Perform Clustering on new representations (KMeans, MeanShift)
* Automatic tune Hyperparameters based on Sillhouette Score
* Construct Clustering Rationale using Statistical Feature Selection
* Processes 2000 votes in <0.1 seconds using heavy vector parallelization
