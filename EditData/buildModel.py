import numpy as np
import pandas as pd
col = [0]
datas =pd.read_excel(r'/home/dang/Desktop/demo1.xlsx',uescol=col)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(datas['Comments'])
new_datas=[]
print(train_vectors)

import time
from sklearn import svm
from sklearn.metrics import classification_report

classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['Label'])



