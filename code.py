import os
import re
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import RandomizedPCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import ward, dendrogram
plt.style.use('ggplot')

pca = RandomizedPCA()
lemmatizer = WordNetLemmatizer()
model = KMeans()
vect = TfidfVectorizer()


folder = '/home/rushil83/Downloads/patent'
filename=[]
data = []

def formating(text):
    t = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    m = []
    for ti in t:
        if re.search('[a-zA-Z]', ti):
            m.append(ti)
    lema = [lemmatizer.lemmatize(t.lower()) for t in m]
    return lema

for files in os.listdir(folder):
    filename.append(files)
    d = open('/home/rushil83/Downloads/patent/' + str(files), 'r').read()
    data.append(d)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8,min_df=0.2, stop_words='english',use_idf=True,tokenizer=formating, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(data)
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
pca = RandomizedPCA(n_components=2).fit_transform(dist)
x = pca[:,0]
y = pca[:,1]

plt.scatter(x,y)
plt.show()

linkage_matrix = ward(dist)

plt.figure(figsize=(25, 10))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')

dendrogram(linkage_matrix,leaf_rotation=90.,  leaf_font_size=8.,truncate_mode='lastp',p = 15)
plt.show()

from scipy.cluster.hierarchy import fcluster
clusters = fcluster(linkage_matrix, 30, criterion='distance')

list = Counter(clusters)
cluster = np.array(clusters).tolist()

def duplicates(lst, item):
  return [i for i, x in enumerate(lst) if x == item]

dict = {}
for i in range(1,5):
    for j in duplicates(cluster,i):
         k=formating(data[j])
         k = [w for w in k if not w in stopwords.words('english')]
         dict[str(i)] = [word for word, word_count in Counter(k).most_common(15)]

print(list)
print(dict)

""" ### printing filename correspond to its cluster
l1 = []
for i in range(4,5):
    k=duplicates(cluster,i)
    for j in k:
        l1.append(filename[j])
print(l1)
"""