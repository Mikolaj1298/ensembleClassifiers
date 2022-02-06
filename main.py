import random

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


class ClfElement:
    def __init__(self, name, classifier):
        self.name = name
        self.classifier = classifier

# funkcja zwracająca zbiór danych zawierający wyłącznie k najbardziej istotnych cech
def selectFeatures(X, y, k, show_info=False):
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)

    # możliwość wyświetlenia informacji o rankingu cech
    if show_info:
        names = X.columns.values[selector.get_support()]
        scores = selector.scores_[selector.get_support()]
        names_scores = list(zip(names, scores))
        ns_df = pd.DataFrame(data=names_scores, columns=['Feat_names', 'F_Scores'])
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending=[False, True])
        print(ns_df_sorted)

    return X_new

# funkcja wykonująca trenowanie modelu oraz
# sprawdzenie jego jakości, zwraca tablicę wyników dla odpowiednich metryk
def checkAccuracy(X, Y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)

    clf.fit(X_train, y_train.values.ravel())
    predictions = clf.predict(X_test)
    return [accuracy_score(y_test, predictions), f1_score(y_test, predictions),
            recall_score(y_test, predictions, zero_division=1),
            precision_score(y_test, predictions, zero_division=1)]


# funkcja przygotowująca eksperyment na podstawie zbioru danych, wybranego klasyfikatora
# oraz ilości powtórzeń
def prepareExperiment(data, clf, iterations, features_number, is_ensemble_random=False):
    y = data["Risk_Flag"]
    X = data.drop("Risk_Flag", axis=1)
    Y = pd.DataFrame(y)

    X_new = selectFeatures(X, y, features_number)
    resultArray = [
        [], [], [], []
    ]

    # główna pętla ekspetymentu
    for i in range(iterations):
        # losowanie klasyfikatora bazowego w przypadku tworzenia klasyfikatora kombinowanego
        if is_ensemble_random:
            modifyEnsembleClf(clf)
        print(clf.classifier)
        for j in range(len(resultArray)):
            result = checkAccuracy(X_new, Y, clf.classifier)
            if result[j] != 0.0 and result[j] != 1.0:
                resultArray[j].append(result[j])
    # zwrócenie tablicy z uśrednionymi wynikami dla każdej z metryk
    return [np.mean(result).round(4) for result in resultArray]

# ustawienie klasyfikatora bazowego na wylosowany
def modifyEnsembleClf(clf):
    offset = 1
    if clf.name == 'AdaBoost':
        offset = 2
    clf.classifier.base_estimator = basicClfArray[random.randint(0, len(basicClfArray) - offset)].classifier

# macierz ilości najbardziej istotnych cech
featureNumberArray = [3, 6, 10]

# macierz klasyfikatorów bazowych
basicClfArray = [
    ClfElement('Naive Bayes', GaussianNB()),
    # ClfElement('MLP', MLPClassifier(random_state=1, max_iter=3)),
    ClfElement('Decision Tree', DecisionTreeClassifier(random_state=0)),
    ClfElement('KNeighbours', KNeighborsClassifier(n_neighbors=3)),
]

# macierz klasyfikatorów kombinowanych
ensembleClfArray = [
    ClfElement('Bagging Classifier',
               BaggingClassifier(n_estimators=10, random_state=0)),
    ClfElement('AdaBoost',
               AdaBoostClassifier(n_estimators=5, random_state=0)),
    ClfElement('Random Forest',
               BaggingClassifier(n_estimators=10, random_state=0)),
]

# macierz klasyfikatorów kombinowanych wykorzystujących głosowanie
ensembleVotingClfArray = [
    ClfElement('Voting (hard)', VotingClassifier(estimators=[
        (basicClfArray[0].name, basicClfArray[0].classifier),
        (basicClfArray[1].name, basicClfArray[1].classifier),
        (basicClfArray[2].name, basicClfArray[2].classifier),
    ], voting='hard')),
    ClfElement('Voting (soft)', VotingClassifier(estimators=[
        (basicClfArray[0].name, basicClfArray[0].classifier),
        (basicClfArray[1].name, basicClfArray[1].classifier),
        (basicClfArray[2].name, basicClfArray[2].classifier),
    ], voting='soft')),
    ClfElement('Voting (soft - weights)', VotingClassifier(estimators=[
        (basicClfArray[0].name, basicClfArray[0].classifier),
        (basicClfArray[1].name, basicClfArray[1].classifier),
        (basicClfArray[2].name, basicClfArray[2].classifier),
    ], voting='soft', weights=[1, 1, 2])),

]

if __name__ == '__main__':
    # wczytywanie zbioru danych

    data = pd.read_csv('./data/Training_Data.csv')
    data = data.drop("Id", axis=1)
    data = data.drop("CITY", axis=1)
    catCols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'STATE']

    # data = pd.read_csv('./data/adult-all.csv')
    # catCols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
    #            'native-country', 'Risk_Flag']

    # kodowanie kolumn zawierających kategorie
    en = LabelEncoder()
    for cols in catCols:
        data[cols] = en.fit_transform(data[cols])

    # eksperyment + zapisywanie danych do pliku
    with open("./results/results_adultIncomeAll.txt", 'w') as f:
        # badanie klasyfikatorów bazowych
        for clf in basicClfArray:
            for featureNumber in featureNumberArray:
                print(clf.name, featureNumber)
                f.write(f'[Accuracy, F1-score, Recall, Precision] for features: {featureNumber}\n')
                f.write(f'{clf.name}, {prepareExperiment(data, clf, 1, featureNumber)}\n')
            f.write("\n")

        # badanie klasyfikatorów kombinowanych
        for clf in ensembleClfArray:
            for featureNumber in featureNumberArray:
                print(clf.name, featureNumber)
                f.write(f'[Accuracy, F1-score, Recall, Precision] for features: {featureNumber}\n')
                f.write(f'{clf.name}, {prepareExperiment(data, clf, 20, featureNumber, is_ensemble_random=True)}\n')
            f.write("\n")

        # badanie klasyfikatorów kombinowanych wykorzystujących głosowanie
        for clf in ensembleVotingClfArray:
            for featureNumber in featureNumberArray:
                print(clf.name, featureNumber)
                f.write(f'[Accuracy, F1-score, Recall, Precision] for features: {featureNumber}\n')
                f.write(
                    f' VotingClassifier (soft) {clf.name}, {prepareExperiment(data, clf, 20, featureNumber)}\n')
            f.write("\n")
