import rampwf as rw

import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Data challenge Absenteism at work'

int_to_cat = {
    0: "Unknown",

    # Group 1: Infectious, Neoplastic, and Immune Diseases
    1: "Infectious, Neoplastic, and Immune Diseases",
    2: "Infectious, Neoplastic, and Immune Diseases",
    3: "Infectious, Neoplastic, and Immune Diseases",

    # Group 2: Chronic and Metabolic Conditions
    4: "Chronic and Metabolic Conditions",
    9: "Chronic and Metabolic Conditions",
    10: "Chronic and Metabolic Conditions",
    11: "Chronic and Metabolic Conditions",

    # Group 3: Neurological, Psychiatric, and Sensory Disorders
    5: "Neurological, Psychiatric, and Sensory Disorders",
    6: "Neurological, Psychiatric, and Sensory Disorders",
    7: "Neurological, Psychiatric, and Sensory Disorders",
    8: "Neurological, Psychiatric, and Sensory Disorders",

    # Group 4: Musculoskeletal, Dermatological, and Genitourinary Conditions
    12: "Musculoskeletal, Dermatological, and Genitourinary Conditions",
    13: "Musculoskeletal, Dermatological, and Genitourinary Conditions",
    14: "Musculoskeletal, Dermatological, and Genitourinary Conditions",
    15: "Musculoskeletal, Dermatological, and Genitourinary Conditions",

    # Group 5: Injuries, External Causes, Pregnancy, and Other Conditions
    16: "Injuries, External Causes, Pregnancy, and Other Conditions",
    17: "Injuries, External Causes, Pregnancy, and Other Conditions",
    18: "Injuries, External Causes, Pregnancy, and Other Conditions",
    19: "Injuries, External Causes, Pregnancy, and Other Conditions",
    20: "Injuries, External Causes, Pregnancy, and Other Conditions",
    21: "Injuries, External Causes, Pregnancy, and Other Conditions",

    # Group 6: Non-Disease Absences (Administrative & Follow-up)
    22: "Non-Disease Absences",
    23: "Non-Disease Absences",
    24: "Non-Disease Absences",
    25: "Non-Disease Absences",
    26: "Non-Disease Absences",
    27: "Non-Disease Absences",
    28: "Non-Disease Absences"
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


from sklearn.model_selection import StratifiedShuffleSplit

def get_cv(X, y):
    """Returns a StratifiedShuffleSplit cross-validator object for use in RAMP workflow."""
    return StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=57)


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
