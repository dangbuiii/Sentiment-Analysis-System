from builtins import dict

import pandas as pd
import string
from underthesea import word_tokenize
col = [0]
datas = pd.read_excel(r'/home/dang/Desktop/demo1.xlsx',
                     usecols=col)
dataList = datas.values.tolist()
word_tk =[]

# texts = [[word for word in document.lower().split() if word not in stopwords]
#          for document in dataList]
# print(texts)
for i in range(0,len(dataList)):
    radars_string = '\n'.join(s for s in dataList[i])
    word_tk.append(word_tokenize(radars_string))

stopwords = set('thì để này cũng bởi chưa cùng đã đang do đó nên nếu cho là được cái quá rất chỉ th'.split(' '))
new_data = [[word for word in document if word not in stopwords] for document in word_tk]
print(new_data)
large = []
for i in range(0,len(new_data)):
    large.append(len(new_data[i]))
print(large)
text = [item for e in new_data for item in e]
print(text)
old= []
e = 0
for i in range(0,len(large)):
    old.append(text[e:large[i]+e])
    e = large[i]+e
new = [' '.join(l) for l in old]
print(new)

dataNew = pd.DataFrame(new)
export_excel = dataNew.to_excel(r'/home/dang/Desktop/DEMO.xlsx')


# set_=  [{item for e in new_data for item in e}]


# from gensim import corpora
#
# dictionary = corpora.Dictionary(new_data)

# print(dictionary.token2id)



#---------------------------------------------------------------------

# bow_corpus = [dictionary.doc2bow(data) for data in new_data]
#
# print(bow_corpus)
#






# from gensim import models
# tfidf = models.TfidfModel(bow_corpus)
# new = []
#
# for i in range(0,len(new_data)):
#     newdata = tfidf[dictionary.doc2bow(new_data[i])]
#     new.append(newdata)
#
#
# print(new)
# data_vector = pd.DataFrame(new)

# export_CSV = data_vector.to_csv(r'/home/dang/Desktop/data_tfidf.csv')

# from sklearn.feature_extraction.text import TfidfVectorizer
# Tfidf_vect = TfidfVectorizer()
# for i in range(0,len(new_data)):
#     Test_X_Tfidf = Tfidf_vect.transform(new)
# print(Test_X_Tfidf)

