import rampwf as rw
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

problem_title = 'Data challenge Absenteism at work'

_prediction_label_names = ["Unknown", 
                           "Infectious, Neoplastic, and Immune Diseases",
                           "Chronic and Metabolic Conditions",
                           "Neurological, Psychiatric, and Sensory Disorders",
                           "Musculoskeletal, Dermatological, and Genitourinary Conditions",
                           "Injuries, External Causes, Pregnancy, and Other Conditions",
                           "Non-Disease Absences"]

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
    """Return an iterable cross-validation split using StratifiedKFold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
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
