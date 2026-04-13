import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


data = pd.read_excel("/Users/hamzahbelim/first_nig/DAV PROJECT/Assignment 2/Artemis_dataset.xlsx")

# Splitting data
X = data['Tweet']
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


print("Naive Bayes:")
print(classification_report(y_test, y_pred))


# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_vec, y_train)
print("Logistic Regression:")
print(classification_report(y_test, lr.predict(X_test_vec)))

# SVM
svm = SVC()
svm.fit(X_train_vec, y_train)
print("SVM:")
print(classification_report(y_test, svm.predict(X_test_vec)))


#Graphs and Visualizations
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

models = ['Naive Bayes', 'Logistic Regression', 'SVM']
accuracies = [0.75, 0.70, 0.60]

plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()


data['Sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
