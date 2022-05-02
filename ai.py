import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

Tumor_data = pd.read_csv("Tumor Cancer Prediction_Data (2).csv")
# Cleaning Data From Null and duplicated Values
Tumor_data.dropna(inplace=True)
Tumor_data.drop_duplicates(inplace=True)

Tumor_data['diagnosis'] = Tumor_data['diagnosis'].replace(to_replace=['B', 'M'], value=[0, 1])
data = Tumor_data.iloc[:, 1:31].values
target = Tumor_data['diagnosis']
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.25, random_state=0)


def svmClassify():
    classify = svm.SVC(kernel='linear')
    classify.fit(X_train, Y_train)
    y_predict = classify.predict(X_test)
    Accuracy = metrics.accuracy_score(Y_test, y_predict)
    print("SVM:")
    print("Accuracy:", Accuracy)
    print("Precision:", metrics.precision_score(Y_test, y_predict))
    print("Recall:", metrics.recall_score(Y_test, y_predict))


# Logistic Regression
def logisticRegression():
    logistic_regression_model = LogisticRegression(solver='liblinear')
    logistic_regression_model.fit(X_train, Y_train)
    y_predict = logistic_regression_model.predict(X_test)
    score = logistic_regression_model.score(X_train, Y_train)
    cnf_matrix = metrics.confusion_matrix(Y_test, y_predict)
    report = classification_report(Y_test, y_predict)
    print("Logistic Regression:")
    print("score:", score)
    print("cnf:", cnf_matrix)
    print("report:", report)


# decision tree
def decisionTree():
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, Y_train)
    Y_predict = decision_tree_model.predict(X_test)
    print("TREE:")
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_predict))


def display():
    while True:
        print("--------------------------------------------")
        print("\nChoose Way of Classification\n"
              " 1-SVM Classification\n"
              " 2-Logistic Regression\n"
              " 3-Decision Tree\n"
              " 4-Exit\n")
        choice = input()
        if choice == '1':
            svmClassify()
        elif choice == '2':
            logisticRegression()
        elif choice == '3':
            decisionTree()
        elif choice == '4':
            break
        else:
            print("Please Enter a Valid Number")


display()