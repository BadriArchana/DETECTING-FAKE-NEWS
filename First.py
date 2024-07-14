import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
news_data = pd.read_csv("news.csv")
print(news_data.head(10))
news_data.info()
print(news_data.shape)
news_data["label"].value_counts()
labels = news_data.label
print(labels.head(10))
x_train, x_test, y_train, y_test = train_test_split(news_data["text"], labels, test_size=0.4, random_state=7)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)
from sklearn.linear_model import PassiveAggressiveClassifier
# Building the PassiveAggressiveClassifier
passive = PassiveAggressiveClassifier(max_iter=50)
passive.fit(tfidf_train, y_train)
y_pred = passive.predict(tfidf_test)
matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(matrix)
sns.heatmap(matrix, annot=True)
plt.show()
Accuracy = accuracy_score(y_test, y_pred)
print(Accuracy * 100)
Report = classification_report(y_test, y_pred)
print(Report)
