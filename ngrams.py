import pickle
import nltk
import pandas as pd
import numpy as np
import category_encoders as ce

with open('C:/Users/Admin/PycharmProjects/CodeMixing/Files/just_words.txt', 'rb') as f:
    just_words = list(pickle.load(f))

with open('C:/Users/Admin/PycharmProjects/CodeMixing/Files/just_tags.txt', 'rb') as f:
    just_tags = list(pickle.load(f))

print(len(just_words))

print(len(np.unique(just_words)))
def generate_ngrams(just_words):
    n_grams_list = []
    for i,word in enumerate(just_words):
            for ngram in nltk.bigrams(word):
                n_grams_list.append(ngram)
            for ngram in nltk.trigrams(word):
                n_grams_list.append(ngram)
    freq_ngrams = nltk.FreqDist(n_grams_list)
    return freq_ngrams.most_common(250)

def gen_features():
    freq_ngrams = generate_ngrams(just_words)
    features = {}
    fields = []
    print(freq_ngrams)
    for i in freq_ngrams:
        i = str("".join(i[0]))
        fields.append(i)
    print(fields)

    count = 0
    features = {}
    for num,j in enumerate(just_words):
        features[num] = {}
        for i in fields:
            if i in j:
                features[num][i] = 1
            else:
                features[num][i] = 0

        features[num]['length'] = len(j)
        features[num]['caps'] =  1 if j[0].isupper() else 0
        features[num]['num_caps'] = sum([True for a in  j if a.isupper()])
        features[num]['isdigit'] = 1 if j.isdigit() else 0
        features[num]['suffixes'] = j[-3:]
        features[num]['target'] = just_tags[num]

    feat = pd.DataFrame.from_dict(features,
                           orient="columns")

    feat_transpose = feat.T
    #print(feat_transpose)
    from sklearn import preprocessing
    df_binary = preprocessing.LabelEncoder()
    feat_transpose['suffixes'] = df_binary.fit_transform((feat_transpose['suffixes']))
    print(feat_transpose.columns)
    #feat_transpose.to_csv('finalngram.csv', index=False, encoding='utf-8')

    #feat_transpose.to_csv("ngram.csv", sep='\t', encoding='utf-8')'''

gen_features()