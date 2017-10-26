
# Clustering
Document clustering with the help of natutal language processing , here we will try to cluster the large database of documents into some common grounds or say topics in the form of clusters

 Performing clustering on a scatter data points ,would give us much lower accuracies compared to data points spreaded in small cluster. The task is to create the cluster by breaking down our document files into words. hence Hierarchical clustering model will be best fitted.
 
 This model of Hierarchical clustering is performed in multi dimensional space hence creating a lot of space complexity as well as time complexity in our model. Better to perform in chunks with the help of pickles(recommended)

 Cluster number can be changed, ideal according to hierarchical clustering in this model it should be 2 as according to the max space distance methodology but cluster number can be vary


## Dataset

Nearly 900 Documents of several topics are given. Every Documents is different from each other and with number of lines in Documents varying from 25 lined to 250 lines. Each Document contain the text belongs to some specific topic such as Science, Sports, and etc.


Dataset representaion in 2D after applying `PCA`(Dimension Reduction).

#### PCA SCATTERPLOT

![PCA SCATTERPLOT](https://github.com/rushil83/document_based_clustering/blob/master/pca_scatter_plot.png)


### Features

- Trained model for hierarchical clustering
- Lemmatizer and tokenizer to get the proper dimension of the word in vector space
- Tfidf vectorizer used to project the words of the documents into the vector space
- Counter used to get the most common element or tag in that particular cluster

## Dependencies used

- import os
- import re
- from collections import Counter
- import numpy as np
- from nltk.tokenize import word_tokenize
- from nltk.tokenize import sent_tokenize
- from nltk.stem import WordNetLemmatizer
- from sklearn.cluster import KMeans
- from sklearn.externals import joblib
- from sklearn.feature_extraction.text import TfidfVectorizer
- from sklearn.decomposition import RandomizedPCA
- import matplotlib.pyplot as plt
- from nltk.corpus import stopwords
- from scipy.cluster.hierarchy import ward, dendrogram


## Results

### Result Dendrogram

![DENDROGRAM](https://github.com/rushil83/document_based_clustering/blob/master/result_dendrogram.png)


Results can vary as taking max space distance into consideration if we take distance as 30 we
can see that in dendrogram ,horizontal line( at d=30 ) in graph cut-cross 4 vertical points hence
4 clusters, similarly various number of cluster can be taken depending upon which distance line
chosen.


#### Taking threshold distance as 30 we got 4 clusters :

-  1: 224 (cluster no 1 contains 224 documents)
-  2: 267 (cluster no 2 contains 267 documents)
-  3: 162 (cluster no 3 contains 162 documents) 
-  4: 243 (cluster no 4 contains 243 documents)



#### Tags/Common elements of each Clusters


- 1 : 'data', 'wavefield', 'survey', 'one', 'dip', 'least', 'seismic', 'structure', 'implementation','x', 'plane',        'using', 'according', 'subsurface', 'multicomponent'

- 2 : 'de', 'moment', 'event', 'seismic', 'magnitude', 'sismique', 'la', 'evenements','formation', "d'un", 'sont',        'dans', 'un', 'le', 'partir'

- 3 : 'stress', 'effective', 'figure', 'pressure', 'pore', 'data', 'velocity', 'computer','environment',                  'component', 'geomechanical', 'method', 'target', 'system', 'mean'

- 4 : 'source', 'element', 'side', 'sound', 'emitting', 'surface', 'acoustic', 'de', 'slide', 'unit','drive', 'e',        'one', 'rod', 'r'



