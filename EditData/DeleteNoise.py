import pandas as pd
import collections
import string
import xlrd
import re
col = [0]
data = pd.read_excel(r'/home/dang/Downloads/Data.xlsx',
                     usecols=col)
dataList = data['Comments'].values.tolist()
# print(dataList)
texts=[]
large = []
# new_data = []
# print(dataList)
for data in dataList:
    comment = data.lower().split(' ')
    texts.append(comment)
print(texts)
# Tách từng từ một cho vào 1 List
text = [item for e in texts for item in e]
print(text)

for e in  text[:]:
    text.remove(e)
    e = e.translate(str.maketrans('', '', string.punctuation))
    e = re.sub(r'\d+','',e)
    e.strip()
    text.append(e)
# #
for i in range(0,len(texts)):
    large.append(len(texts[i]))
#
#
#
# -------------------------------------------------
#
wrongWords1= ['dc','đk','dk']
wrongWords2= ['k','ko','hok']
for i in range(0,len(text)):
    if text[i] == 'vs':
        text[i] = 'với'
    # elif text[i] == 'khúc tầm trung':
    #     text[i] = 'khúc'
    # elif text[i] == 'cũng đáng đồng':
    #     text[i] = 'đáng'
    for j in range(0,len(wrongWords1)):
        if text[i]==wrongWords1[j]:
            text[i] = 'được'
    for h in range(0,len(wrongWords2)):
        if text[i]== wrongWords2[h]:
            text[i] = 'không'
#
old= []
e = 0
for i in range(0,len(large)):
    old.append(text[e:large[i]+e])
    e = large[i]+e
new = [' '.join(l) for l in old]
print(new)
# # # #
# # #
dataNew = pd.DataFrame(new)
export_excel = dataNew.to_excel(r'/home/dang/Desktop/demo1.xlsx')

#
# #--------------------------------------------------
#  # BUILD LIBRARY
#
#
# frequency = collections.defaultdict(int)
# for i in text:
#         frequency[i] +=1
# print(sorted(frequency.keys()))
# dic_data = pd.DataFrame(sorted(frequency.keys()))
# export_excel = dic_data.to_excel(r'/home/dang/Desktop/dic_demo1.xlsx')
# processed_corpus = [[token for token in text if frequency[token]>=0]
#                     for text in new_dataList]
# from gensim import corpora
# dictionary = corpora.Dictionary(processed_corpus)
# print(dictionary)




#
# # output = [[token for token in text if frequency[token] > 1] for text in texts]
# print(frequency)
#-----------------------------------------------------
# string = str(dataList[10])
# newString = string.replace('đk','được')
# print(newString)






