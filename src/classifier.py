from sklearn.svm import SVC
from src.DataSet import DataSet
import pickle
import os

class Classifier():

    def train(self, X_train, y_train):
        # Use a linear SVC
        svc = SVC(C=10, kernel='rbf')
        # Check the training time for the SVC
        svc.fit(X_train, y_train)
        self.classifier = svc
        return svc

    def predict(self, X, n_predict=1):
        return self.classifier.predict(X[0:n_predict])


if __name__ == "__main__":
    if not os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data.p")):
        dataset = DataSet()
        X_train, y_train, X_scaler = dataset.prepare_data_set()
        pickle.dump([X_train, y_train, X_scaler], open("data.p", "wb"))
    else:
        X_train, y_train, X_scaler = pickle.load(open("data.p", "rb"))

    if not os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "classifier.p")):
        cls = Classifier()
        cls.train(X_train, y_train)
        pickle.dump(cls, open("classifier.p", "wb"))
    else:
        cls = pickle.load(open("classifier.p", "rb"))

    print("predicted: ", cls.predict(X_train))
    print("original: ", y_train[0])
