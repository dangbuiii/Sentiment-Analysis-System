# a = ['a b c','d e g','h j k']
import string
import re
# b= []
# c=[]
# for word in a:
#     words = word.split(' ')
#     b.append(words)
# print(b)
# text = [item for e in b for item in e]
# for word in b:
#     c.append(len(word))
# print(c)
# old= []
#
# e = 0
# for i in range(0,len(c)):
#     old.append(text[e:c[i]+e])
#     e = c[i]+e
#
#
#
# #
# print(old)
# new = [' '.join(l) for l in old]
# print(new)

# for i in text:

# space = ' '
# string2 = space.join(text)
# a = 'Hello@ wo$%rld 123'
# b= a.translate(str.maketrans('', '', string.punctuation))
# b = re.sub(r'\d+','',b)
# print(b)
# import numpy as np
# from sklearn.svm import SVC
# X1 = [[1,3], [3,3], [4,0], [3,0], [2, 2]]
# y1 = [1, 1, 1, 1, 1]
# X2 = [[0,0], [1,1], [1,2], [2,0]]
# y2 = [-1, -1, -1, -1]
# X = np.array(X1 + X2)
# y = y1 + y2
#
# clf = SVC(kernel='linear', C=1E10)
# clf.fit(X, y)
# print (clf.support_vectors_)
import numpy as np

# from sklearn import svm
# X = [[0,0],[1,1]]
# y = [0,1]
# clf = svm.SVC(gamma='scale')
# print(clf.predict([[2., 2.]]))

# import pandas as pd
#
# datas = pd.read_excel(r'/home/dang/Desktop/demo1.xlsx')
#
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# # Create feature vectors
# vectorizer = TfidfVectorizer(min_df = 5,
#                              max_df = 0.8,
#                              sublinear_tf = True,
#                              use_idf = True)
# train_vectors = vectorizer.fit_transform(datas['Comments'])
# # print(train_vectors)
#
# import time
# from sklearn import svm
# from sklearn.metrics import classification_report
# from  sklearn.linear_model import
# # Perform classification with SVM, kernel=linear
# classifier_linear = svm.SVC(kernel='linear')
# t0 = time.time()
# classifier_linear.fit(train_vectors, datas['Label'])
# t1 = time.time()
# prediction_linear = classifier_linear.predict(test_vectors)
# t2 = time.time()
# time_linear_train = t1-t0
# time_linear_predict = t2-t1
# print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
# report = classification_report(testData['Label'], prediction_linear, output_dict=True)
# print('positive: ', report['pos'])
# print('negative: ', report['neg'])
#------------------------------------------------------------------------------------------------------
# import pandas as pd
# from sklearn.datasets import load_iris
# iris = load_iris()
#
# print(dir(iris))
#
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['target'] = iris.target
# from matplotlib import pyplot as plt
#
# df0 = df[df.target==0]
# df1 = df[df.target==1]
# df2 = df[df.target==2]
# plt.show(plt.scatter(df0['sepal length (cm)'], df1['sepal width (cm)'],color = 'green',marker='+'))


# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# from sklearn.svm import SVC
# clf = SVC(gamma='auto')
# clf.fit(X, y)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# print(clf.predict([[-0.8, -1]]))

# docA = "bây giờ mận mới hỏi đào"
# docB = "vườn hồng có lối ai vào hay chưa"
#
# wordsA = docA.split()
# wordsB = docB.split()
#
# wordSet = set(wordsA).union(set(wordsB))
#
# wordDictA = dict.fromkeys(wordSet, 0)
# wordDictB = dict.fromkeys(wordSet, 0)
#
# for word in wordsA:
#     wordDictA[word]+=1
#
# for word in wordsB:
#     wordDictB[word] += 1
#
# import pandas as pd
# pd.DataFrame([wordDictA, wordDictB])
#
# def computeTF(wordDict, words):
#     tfDict = {}
#     wordsCount = len(words)
#     for word, count in wordDict.items():
#         tfDict[word] = count/float(wordsCount)
#     return tfDict
#
# tfdocA = computeTF(wordDictA, wordsA)
# tfdocB = computeTF(wordDictB, wordsB)
#
#
#
# def computeIDF(docList):
#     import math
#     idfDict = {}
#     N = len(docList)
#
#     idfDict = dict.fromkeys(docList[0].keys(), 0)
#     for doc in docList:
#         for word, val in doc.items():
#             if val > 0:
#                 idfDict[word] += 1
#
#     for word, val in idfDict.items():
#         idfDict[word] = math.log10(N / float(val))
#
#     return idfDict
# idfs = computeIDF([wordDictA, wordDictB])
#
# def computeTFIDF(tfDocs, idfs):
#     tfidf = {}
#     for word, val in tfDocs.items():
#         tfidf[word] = val*idfs[word]
#     return tfidf
#
# tfidfDocA = computeTFIDF(tfdocA, idfs)
# tfidfDocB = computeTFIDF(tfdocB, idfs)
#
# import pandas as pd
# print(pd.DataFrame([tfidfDocA, tfidfDocB]))

import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
print(X)



