from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
g_train_acc = accuracy_score(y_train, gnb.predict(X_train))
g_test_acc = accuracy_score(y_test, gnb.predict(X_test))

X_train_bin = (X_train > X_train.mean()).astype(int)
X_test_bin = (X_test > X_train.mean()).astype(int)

bnb = BernoulliNB()
bnb.fit(X_train_bin, y_train)
b_train_acc = accuracy_score(y_train, bnb.predict(X_train_bin))
b_test_acc = accuracy_score(y_test, bnb.predict(X_test_bin))

print("GaussianNB Training Accuracy:", g_train_acc)
print("GaussianNB Test Accuracy:", g_test_acc)
print("BernoulliNB Training Accuracy:", b_train_acc)
print("BernoulliNB Test Accuracy:", b_test_acc)