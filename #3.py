import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Sample data
sentences = ["This is a sentence.", "Another sentence here.", "Yet another sentence. Dr. Smith works at Inc. The price increased by 4.3%."]
labels = [1, 1, 0]  # 1 for end of sentence, 0 for not

# Tokenize sentences into words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Decision Tree model
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))

# Support Vector Machine model
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
print("Support Vector Machine Accuracy:", accuracy_score(y_test, svm_pred))

# Logistic Regression model
logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
logreg_pred = logreg_clf.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))

# Neural Network model
mlp_clf = MLPClassifier(max_iter=1000)
mlp_clf.fit(X_train, y_train)
mlp_pred = mlp_clf.predict(X_test)
print("Neural Network Accuracy:", accuracy_score(y_test, mlp_pred))

# Example usage for sentence segmentation
test_sentence = "Yet another sentence. Dr. Smith works at Inc. The price increased by 4.3%."
test_sentence_vectorized = vectorizer.transform([test_sentence])
prediction = mlp_clf.predict(test_sentence_vectorized)[0]
if prediction == 1:
    print("End of sentence.")
else:
    print("Not end of sentence.")
