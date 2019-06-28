import pickle
import nltk
import pandas as pd
import numpy as np
from wordcloud import WordCloud

with open('C:/Users/Admin/PycharmProjects/CodeMixing/Files/just_words.txt', 'rb') as f:
    just_words = list(pickle.load(f))

with open('C:/Users/Admin/PycharmProjects/CodeMixing/Files/just_tags.txt', 'rb') as f:
    just_tags = list(pickle.load(f))

words_num = np.unique(just_words)
df = pd.DataFrame({'col':just_words})
#print (df)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

text = " ".join(review for review in df.col)
print(text)

wordcloud = WordCloud(background_color="white", max_words=1000, contour_color='firebrick').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

cfd = nltk.ConditionalFreqDist(
           (num, len(word))
           for num in words_num
           for word in just_words)

plt.figure(figsize=(16,8))
cfd.plot()
plt.show()