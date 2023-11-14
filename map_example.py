from modal import Image, Stub

image = Image.debian_slim().pip_install("scikit-learn~=1.2.2")
stub = Stub()

with image.run_inside():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier


@stub.function(image=image)
def fit_knn(k):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    score = float(clf.score(X_test, y_test))
    print("k = %3d, score = %.4f" % (k, score))
    return score, k


@stub.local_entrypoint()
def main():
    # Do a basic hyperparameter search
    best_score, best_k = max(fit_knn.map(range(1, 100)))
    print("Best k = %3d, score = %.4f" % (best_k, best_score))
