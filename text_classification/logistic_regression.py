import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from text_classification import download_dataset


class LogRegTextClassification(object):
    """
    Run Logistic Regression on CountVectorizer
    texts = List[text]

    """
    def __init__(self):
        self.vectorizer = None
        self.classifier = None

    def fit(self, texts, labels):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(texts)
        x = self.vectorizer.transform(texts)
        self.classifier = LogisticRegression(max_iter=1000, C=0.1)
        self.classifier.fit(x, labels)

    def predict(self, texts):
        x = self.vectorizer.transform(texts)
        proba = self.classifier.predict_proba(x)
        return proba


def main():
    """
    Results in:
    Accuracy on train: 0.7959583973195589
    Accuracy on val: 0.7422660263569354
    Accuracy on test: 0.7386642841188296
    :return:
    """
    df_train, df_val, df_test = download_dataset()
    train_labels = df_train["label"].tolist()
    val_labels = df_val["label"].tolist()
    test_labels = df_test["label"].tolist()
    clf = LogRegTextClassification()
    clf.fit(df_train["text"].tolist(), train_labels)
    for x, y, name in zip(
            [df_train, df_val, df_test],
            [train_labels, val_labels, test_labels],
            ['train', 'val', 'test']
    ):
        proba = clf.predict(x["text"].tolist())
        thr = 0.5
        predicted_class = (proba[:, 1] > thr).astype(int)
        score = accuracy_score(predicted_class, y)
        print("Accuracy on {}: {}".format(name, score))


if __name__ == "__main__":
    main()
