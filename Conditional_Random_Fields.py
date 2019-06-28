import urllib
import json
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import metrics
import matplotlib.pyplot as plt

# Using our data to build a CRF model
link = 'http://dasdipankar.com/ICON_NLP_Tool_Contest_2017/HI-EN.json'
js = urllib.request.urlopen(link)

js_raw = js.readlines()

js_str = [by.decode('utf-8') for by in js_raw]

js_as_json = json.loads(''.join(js_str))

just_text = [i['text'] for i in js_as_json]


def word2features(sent, i):
    word = sent[i]
    #     print(word)
    #     postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #         'postag': postag,
        #         'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1]
        #         postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            #             '-1:postag': postag1,
            #             '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        #         postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            #             '+1:postag': postag1,
            #             '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

just_text_words = [str(i).split() for i in just_text]

wt_sentences = [str(i['lang_tagged_text']).strip() for i in js_as_json]

wts_split = [wts.split(' ') for wts in wt_sentences]

only_sent = []
only_tags = []
for wts in wts_split:
    only_sent.append([i.split('\\')[0] for i in wts if '\\' in i])
    only_tags.append([i.split('\\')[1] for i in wts if '\\' in i])

hien_train = [sent2features(s) for s in only_sent]

en_repl = set(['E', 'EM', 'EN:', 'ENEOS', 'HEN#', 'en', 'NE', 'RN', 'MIX', 'ENC', 'EN,'])
hi_repl = set(['HII', 'HIS','USN'])
un_repl = set(['U', 'UI', 'UN.', 'UNEOS', '_', 'uN', 'un','', 'MI'])
all_repl = [en_repl, hi_repl, un_repl]
all_repl_n = ['en_repl', 'hi_repl', 'un_repl']
to_repl = {'en_repl':'EN', 'hi_repl':'HI', 'un_repl':'UN'}

for j,lang in enumerate(all_repl):
    for t in only_tags:
        for i,tag in enumerate(t):
            if tag in lang:
                t[i] = to_repl[all_repl_n[j]]

# train test split
X_train, X_test, y_train, y_test = train_test_split(hien_train, only_tags, test_size=0.20, random_state=42)

import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

classification_report = metrics.flat_classification_report(y_test,y_pred)
print(classification_report)

print("Accuracy score",metrics.flat_accuracy_score(y_test,y_pred))

labels = ['HI','EN','UN','EMT']

# define fixed parameters and parameters to search
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.1),
    'c2': scipy.stats.expon(scale=0.01),
}

# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs = rs.fit(X_train, y_train)

# crf = rs.best_estimator_
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('Cv', rs.best_estimator_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

CRF = rs.best_estimator_
ypred = CRF.predict(X_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=labels, digits=3
))


_x = [s['c1'] for s in rs.cv_results_['params']]
_y = [s['c2'] for s in rs.cv_results_['params']]
_c = [s for s in rs.cv_results_['mean_test_score']]

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
    min(_c), max(_c)
))

ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])
plt.show()

print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))


