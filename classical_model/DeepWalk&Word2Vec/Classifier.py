from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class Classifier(object):
    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = clf
        self.label_encoder = MultiLabelBinarizer()

    def evaluate(self, y_true, y_pred):
        average_list = ["micro", "macro", "samples", "weighted"]
        results = {}
        y_true = self.label_encoder.transform(y_true)
        y_pred = self.label_encoder.transform(y_pred)
        for average in average_list:
            results[average] = f1_score(y_true, y_pred, average=average)
        return results
        
    def split_train_evaluate(self, X, Y, test_size=0.2, **kwargs):
        X = [self.embeddings[x] for x in X]
        self.label_encoder.fit(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, **kwargs)
        self.clf.fit(X_train, Y_train)
        y_pred = self.clf.predict(X_test)
        return self.evaluate(Y_test, y_pred)