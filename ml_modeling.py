import pandas as pd
import numpy as np
import math
import os


from sdv.tabular import CTGAN, CopulaGAN, GaussianCopula, TVAE
from sdv.tabular.base import BaseTabularModel
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score
from sklearn.metrics import roc_curve, auc
from utils import get_logger
from glob import glob

log = get_logger("general", "ml_modeling.log")


def calc_n_samples(n_majority, n_minority, percent_of_majority):
    return math.ceil(((percent_of_majority / 100) * n_majority) - n_minority)


def percent(num):
    return "{0:.2%}".format(num)


def split_data(data: pd.DataFrame):
    removed_cols = ['user', 'day', 'week', 'starttime',
                    'endtime', 'sessionid', 'insider', 'timeind', 'Unnamed: 0']
    x_cols = [i for i in data.columns if i not in removed_cols]

    first_half = data[data.week <= max(data.week)/2].copy()
    second_half = data[data.week > max(data.week)/2].copy()

    np.random.seed(45)

    selectedTrainUsers = set(first_half[first_half.insider > 0]['user'])
    nUsers = np.random.permutation(
        list(set(first_half.user) - selectedTrainUsers))
    trainUsers = np.concatenate(
        (list(selectedTrainUsers), nUsers[:400-len(selectedTrainUsers)]))
    unKnownTestUsers = list(set(second_half.user) - selectedTrainUsers)

    x_train = first_half[first_half.user.isin(trainUsers)][x_cols]
    y_train = first_half[first_half.user.isin(trainUsers)]['insider']
    y_train_binary = y_train > 0

    x_test = second_half[second_half.user.isin(unKnownTestUsers)][x_cols]
    y_test = second_half[second_half.user.isin(unKnownTestUsers)]['insider']
    y_test_binary = y_test > 0

    return (x_train, x_test, y_train_binary, y_test_binary)


def get_training_data(data):
    x_train, x_test, y_train, y_test = split_data(data)

    x_train["insider"] = y_train
    x_test["insider"] = y_test

    return (x_train, x_test)


def calc_scores(y_test, y_pred, details: dict):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_positive_rate = percent(fp / (fp + tn))

    recall = percent(recall_score(y_test, y_pred, pos_label=True),)
    precision = percent(precision_score(y_test, y_pred, pos_label=True),)
    accuracy = percent(accuracy_score(y_test, y_pred))
    f1 = percent(f1_score(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=True)
    auc_s = percent(auc(fpr, tpr))

    synthetic_model_name = details["synthetic_model_name"]
    ml_model_name = details["ml_model_name"]
    train_length = details["train_length"]
    test_length = details["test_length"]
    percent_of_majority = details["percent_of_majority"]

    folder_name = f"./scores/{percent_of_majority}_percent/{synthetic_model_name}/"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    file_name = f"{folder_name}/{ml_model_name}.txt"

    with open(file_name, "w") as file:
        file.writelines([
            ml_model_name.upper(),
            f"\n{synthetic_model_name.upper()}\n{percent_of_majority}% malicious samples",
            f"\n\nRecall\t\t{recall}",
            f"\nPrecision\t{precision}",
            f"\nFPR\t\t{false_positive_rate}",
            f"\nF1\t\t{f1}",
            f"\nAUC\t\t{auc_s}",
            f"\nAccuracy\t{accuracy}",
            f"\n\nTP\t\t{tp}",
            f"\nFP\t\t{fp}"
            f"\nFN\t\t{fn}"
            f"\nTN\t\t{tn}",
            f"\n\n\nTrain Len\t{train_length}",
            f"\nTest Len\t{test_length}"
        ])

    if float(recall[:-1]) > 79.5:
        log.info(
            f"{percent_of_majority}_{synthetic_model_name}_{ml_model_name}_{recall}"[:-1].replace(" ", "-"))


def get_synthetic_model(name, path) -> BaseTabularModel:
    if name == "copulagan":
        model = CopulaGAN
    elif name == "ctgan":
        model = CTGAN
    elif name == "gaussiancopula":
        model = GaussianCopula
    elif name == "tvae":
        model = TVAE
    else:
        raise Exception

    return model.load(path)


ml_models = {
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=0),
    "Random Forest": lambda: RandomForestClassifier(n_jobs=-1, random_state=0),
    "XGBoost": lambda: XGBClassifier(n_jobs=-1, random_state=0),
    "Gaussian NB": lambda: GaussianNB(),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=0),
}


def run(data, percent_of_majority):
    for gen_model_path in glob("./generative_models/*"):
        train_data, test_data = get_training_data(data)

        insider_count = train_data["insider"].value_counts()
        n_samples_to_generate = calc_n_samples(
            insider_count[False], insider_count[True], percent_of_majority)

        x_test = test_data.drop("insider", axis=1)
        y_test = test_data["insider"]

        gen_model_name = os.path\
            .normpath(gen_model_path)\
            .split(os.path.sep)[-1][:-4]
        gen_model = get_synthetic_model(
            gen_model_name, gen_model_path)

        if percent_of_majority == 0:
            x_train = train_data.drop("insider", axis=1)
            y_train = train_data["insider"]
        else:
            # generate artificial malicious samples
            synthetic_malicious_samples = gen_model.sample(
                num_rows=n_samples_to_generate)
            synthetic_malicious_samples = synthetic_malicious_samples.copy()
            synthetic_malicious_samples["insider"] = True

            # balance train dataset
            balanced_train_data = pd.concat(
                [train_data, synthetic_malicious_samples], ignore_index=True)

            x_train = balanced_train_data.drop("insider", axis=1)
            y_train = balanced_train_data["insider"]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        for ml_model_name, model_factory in ml_models.items():

            estimator = model_factory()
            estimator.fit(x_train, y_train)
            predictions = estimator.predict(x_test)

            details = {
                "synthetic_model_name": gen_model_name,
                "ml_model_name": ml_model_name,
                "train_length": len(x_train),
                "test_length": len(x_test),
                "percent_of_majority": percent_of_majority
            }

            calc_scores(y_test, predictions, details)


if __name__ == "__main__":
    data: pd.DataFrame = pd.read_csv(f"./dataset/weekr5.2.csv")

    percent_of_majority = 5

    while percent_of_majority <= 5:
        print(percent_of_majority, "\n---------")

        log.info(
            f"Generate {percent_of_majority} percent majority of minority class".replace(" ", "_"))
        run(data, percent_of_majority)

        percent_of_majority += 5
