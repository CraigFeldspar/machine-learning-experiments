import unicodedata
import re
import gensim
import json
import numpy as np
import matplotlib.pyplot as plt


def remove_accents(text):
    nkfd_form = unicodedata.normalize('NFKD', text)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])


def tokenize(text):
    return re.sub("\s+", " ", re.sub("[^A-Za-z0-9]", " ", remove_accents(text))).split()  # [^A-Za-z0-9'] not lower.


def remove_small_words(list):
    return [word.lower() for word in list if len(word) > 1]


filename = "./wanadev.txt"
with open('wanadev.json') as f:
    data = json.load(f)


#text = (open(filename).read()).lower()
text = [i["title"] + "\n" + i["text"] for i in data]
text_data = [tokenize(item.lower()) for item in text]

size = 0
for el in text_data:
    size += len(el)

print(size)
try:
    model = gensim.models.Word2Vec.load("word2vec.model")  # ..
except IOError:
    model = gensim.models.Word2Vec(text_data, size=40, window=5, workers=8, min_count=2)  # re-train only if required.

model.save("word2vec.model")

print(model.most_similar(positive=["trois", "un", "deux"], topn=10))
print(model.most_similar(positive=["fantastique", "incroyable"], topn=10))
print(model.most_similar(positive=["probleme"], negative=["efficace"], topn=10))
print(model.most_similar(positive=["efficace"], topn=10))
print(model.most_similar(positive=["technique", "complexe"], topn=10))

#word_list = ['3d', 'mon', 'ma', 'son', 'sa', 'a', 'innovation', 'sans', 'et', 'est', 'immersion', 'api', 'integration', 'technique', 'rendu', 'solution', 'wanadev', 'entreprise', 'yannick', 'fabien', 'baptiste', 'pierre', 'leo', 'benjamin', 'docker', 'container', 'moteur', 'manuel', 'come', 'api', 'web', 'webgl', 'vr', 'client', 'projet', 'le', 'la', 'comme', 'avec', 'pour', 'francois']
#word_list = ['3d', 'mon', 'ma', 'son', 'sa', 'technique', 'rendu', 'solution', 'wanadev', 'entreprise', 'yannick', 'fabien', 'baptiste', 'pierre', 'benjamin', 'docker', 'container', 'moteur', 'manuel', 'come', 'api', 'web', 'webgl', 'vr', 'client', 'projet', 'le', 'la', 'comme', 'avec', 'pour']

#for word in word_list:
#    plt.scatter(model.wv[word][0], model.wv[word][1])
#    plt.annotate(word, xy=(model.wv[word][0:2]))

#plt.show()