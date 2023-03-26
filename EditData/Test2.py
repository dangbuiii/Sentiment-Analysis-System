import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


Corpus = pd.read_excel(r'/home/dang/Desktop/DEMO.xlsx',)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Comments'],Corpus['Label'])


from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Corpus['Comments'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)

Test_X_Tfidf = Tfidf_vect.transform(Test_X)


#
#
SVM = svm.SVC(kernel='linear')
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)
# print(Test_X_Tfidf)
def testComment(comment):

    p=comment
    Test_p =Tfidf_vect.transform([p])
    return SVM.predict(Test_p)

    # print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(testComment("Sản phẩm này tuyệt vời"))