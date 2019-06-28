import pickle
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import category_encoders as ce


from sklearn.preprocessing import StandardScaler
ngram = pd.read_csv('final_ngram.csv')
X = ngram.drop('target', axis=1)
y = ngram['target']

X =StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
cols = [i for i in range(2)]
principalDf = pd.DataFrame(data=principalComponents, columns=cols)

finalDf = pd.concat([principalDf,ngram[['target']]],axis =1)
pca_var = pca.explained_variance_ratio_

sum = 0
for i in pca_var:
    sum = sum + i
print(sum)
print(pca.explained_variance_ratio_)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

'''fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['HI', 'EN', 'UN','EMT']
colors = ['r', 'g', 'b','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, cols[0]]
               , finalDf.loc[indicesToKeep, cols[1]]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()'''


pca = PCA(n_components=200)
principalComponents = pca.fit_transform(X)
cols = [i for i in range(200)]
principalDf = pd.DataFrame(data=principalComponents, columns=cols)

finalDf = pd.concat([principalDf,ngram[['target']]],axis =1)
pca_var = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(pca_var)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(200), pca_var, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(200), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()

'''plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()'''

sum = 0
for i in pca_var:
    sum = sum + i
print(sum)
print(pca.explained_variance_ratio_)

from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
train_X, test_X, train_y, test_y = train_test_split( X, y, test_size=1/7.0, random_state=0)
pca.fit(train_X)

import time
start = time.time()
from sklearn.svm import SVC
svclassifier = SVC()
svclassifier.fit(train_X, train_y)
y_pred = svclassifier.predict(test_X)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))
print(accuracy_score(test_y,y_pred))

end = time.time()
print(time.clock())
print(end-start)