import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('pointure.data')
df
df.columns
import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(df['Genre'])
print(encoded_labels)
df['Genre'] = encoded_labels

df
X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]
from sklearn.model_selection import train_test_split


#decomposer les donnees predicteurs en training/testing
testSize = 0.2
randomState = 44
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_naive_bayes1 = gnb.predict(X_train)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
print("Accuracy du modele Naive Bayes predit: " + str(accuracy))


recall_score_train = metrics.recall_score(y_train, y_naive_bayes1)
print("recall score du modele Naive Bayes predit: " + str(recall_score_train))

f1_score_train = metrics.f1_score(y_train, y_naive_bayes1)
print("F1 score du modele Naive Bayes predit: " + str(f1_score_train))
y_naive_bayes2 = gnb.predict(X_test)
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_test.shape[0],(y_test != y_naive_bayes2).sum()))

recall_score_test = metrics.recall_score(y_test, y_naive_bayes2)
print("recall score du modele Naive Bayes predit: " + str(recall_score_test))

f1_score_test = metrics.f1_score(y_test, y_naive_bayes2)
print("F1 score du modele Naive Bayes predit: " + str(f1_score_test))
# Ecrire les metriques et les infos du modele dans un fichier texte
with open("metrics.txt", 'a') as metriquefile:
        metriquefile.write("Accuracy train:  {0:2.1f} \n".format(accuracy))
        metriquefile.write("Recall score train:  {0:2.1f} \n".format(recall_score_train))
        metriquefile.write("F1 score train: {0:2.1f}\n".format(f1_score_train))
        metriquefile.write("Recall score test:  {0:2.1f} \n".format(recall_score_test))
        metriquefile.write("F1 score test: {0:2.1f}\n".format(f1_score_test))

with open("modele.txt", 'a') as modelefile:
        modelefile.write("Test Size:  {0:2.1f} \n".format(testSize))
        modelefile.write("Random State:  {0:2.1f} \n".format(randomState))
        d = {'Taille(cm)':[183], 'Poids(kg)':[59], 'Pointure(cm)':[20]}
dfToPredict = pd.DataFrame(data=d)
dfToPredict
yPredict = gnb.predict(dfToPredict)
