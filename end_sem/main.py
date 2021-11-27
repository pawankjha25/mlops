
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def run_classifier(hyperparameter, run):
    mt_map = {}
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    clf = svm.SVC(gamma=hyperparameter)
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.30, shuffle=False
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_test, y_test, test_size=0.15, shuffle=False
    )
    clf.fit(X_train, y_train)
    predicted_tr = clf.predict(X_train)
    predicted_test = clf.predict(X_test)
    predicted_dev = clf.predict(X_dev)
    train_score = metrics.accuracy_score(y_train, predicted_tr)
    test_score = metrics.accuracy_score(y_test, predicted_test)
    dev_score = metrics.accuracy_score(y_dev, predicted_dev)
    mt_map['train'] = train_score
    mt_map['test'] = train_score
    mt_map['dev'] = train_score
    mt_map['hyperparameter'] = hyperparameter
    mt_map['run_id'] = run
    return mt_map
def main():
    hyperparameters = [0.01, 0.001, 0.0001]
    mt_maps = {}
    for hp in hyperparameters:
        for i in range(1, 4):
            mt_maps[i] = run_classifier(hp, i)
        mt_maps[hp] = mt_maps[i]

    print(mt_maps)


if __name__=="__main__":
    main()