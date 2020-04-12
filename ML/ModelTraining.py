from sklearn import ensemble
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

class Gdb:
    def __init__(self, verbose=1):
        self.gb_estimator = ensemble.GradientBoostingClassifier( criterion='mse', verbose=verbose)

    def train(self, X , y):
        self.gb_estimator.fit(X , y)

    def predict(self, X):
        p = self.gb_estimator.predict_proba(X)
        if (sum(p[0][:]) > sum(p[1][:])):
            return "fake"
        else:
            return "satire"

class Svm:
    def __init__(self):
        self.svm = svm.SVC()

    def train(self, X , y):
        self.svm.fit(X , y)

    def predict(self, X):
        p = self.svm.predict_proba(X)
        if (sum(p[0][:]) > sum(p[1][:])):
            return "fake"
        else:
            return "satire"

class Knn:
    def __init__(self):
        self.knn = KNeighborsClassifier()

    def train(self, X , y):
        self.knn.fit(X , y)

    def predict(self, X):
        p = self.knn.predict_proba(X)
        if (sum(p[0][:]) > sum(p[1][:])):
            return "fake"
        else:
            return "satire"

