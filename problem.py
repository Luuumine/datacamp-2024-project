import rampwf as rw

import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Data challenge Absenteism at work'

int_to_cat = {
    1: "Certain infectious and parasitic diseases",
    2: "Neoplasms",
    3: "Diseases of the blood and blood-forming organs and certain \
        disorders involving the immune mechanism",
    4: "Endocrine, nutritional and metabolic diseases",
    5: "Mental and behavioural disorders",
    6: "Diseases of the nervous system",
    7: "Diseases of the eye and adnexa",
    8: "Diseases of the ear and mastoid process",
    9: "Diseases of the circulatory system",
    10: "Diseases of the respiratory system",
    11: "Diseases of the digestive system",
    12: "Diseases of the skin and subcutaneous tissue",
    13: "Diseases of the musculoskeletal system and connective tissue",
    14: "Diseases of the genitourinary system",
    15: "Pregnancy, childbirth and the puerperium",
    16: "Certain conditions originating in the perinatal period",
    17: "Congenital malformations, deformations and chromosomal abnormalities",
    18: "Symptoms, signs and abnormal clinical and laboratory findings,\
          not elsewhere classified",
    19: "Injury, poisoning and certain other consequences of external causes",
    20: "External causes of morbidity and mortality",
    21: "Factors influencing health status and contact with health services",
    22: "Patient follow-up",
    23: "Medical consultation",
    24: "Blood donation",
    25: "Laboratory examination",
    26: "Unjustified absence",
    27: "Physiotherapy",
    28: "Dental consultation"
}

_prediction_label_names = int_to_cat.values()

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='accuracy', precision=4),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def load_data(path='.', file='X_train.csv'):
    path = Path(path) / "data"
    X_df = pd.read_csv(path / file)

    y = X_df['target']
    X_df = X_df.drop(columns=['target'])

    return X_df, y


# READ DATA
def get_train_data(path='.'):
    file = 'X_train.csv'
    return load_data(path, file)


def get_test_data(path='.'):
    file = 'X_test.csv'
    return load_data(path, file)
