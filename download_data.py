from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path('data')


if __name__ == '__main__':
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    # Load the data
    print('Loading the data...', end='', flush=True)
    target = 'Reason for absence'

    df = pd.read_csv('data/base_data.csv', sep=';')

    df = df.rename(columns={target: 'target'})

    icd_mapping_5 = {

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

    # Apply the mapping to create a grouped category column
    df["target"] = df["target"].map(icd_mapping_5)
    

    X_train, X_test = train_test_split(
        df, test_size=0.2, random_state=57, shuffle=True,
        stratify=df['target']
    )

    # Save the data
    X_train.to_csv(DATA_PATH / 'X_train.csv', index=False)
    X_test.to_csv(DATA_PATH / 'X_test.csv', index=False)
    print('done')
