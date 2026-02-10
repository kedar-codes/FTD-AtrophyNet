import os
import glob
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from imblearn.pipeline import Pipeline # Use imblearn's Pipeline
from imblearn.over_sampling import SMOTE # Import SMOTE
from sklearn.linear_model import LogisticRegression

####################################################################################################
# Functions
####################################################################################################

##################################################
# Classifier Function(s)
##################################################

# Choose desired classifier; add more as needed

#-------------------------------------------------
# Multinomial Logistic Regression (scikit-learn)

def multinomial_log_reg():    
    return Pipeline([
        ("scaler", StandardScaler()), # Standardize features
        ('smote', SMOTE(random_state=42, k_neighbors=2)),
        ("clf", LogisticRegression( # The classifier
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced"
        )),
    ])

#-------------------------------------------------
# Support Vector Machine (scikit-learn)

def support_vector_machine():
    return Pipeline([
    ("scaler", StandardScaler()), # Scale/normalize feature data
    ('smote', SMOTE(random_state=42, k_neighbors=2)), # Apply SMOTE (only on the training folds)
    ("svm", svm.SVC(
        kernel="linear",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

##################################################
# Helper Functions
##################################################

def load_data_map(classes_dirs, controls_dir):

    # Scans directories and returns a DataFrame of subjects
    data = []
    
    # Load subtypes (classes)
    for label, folder in classes_dirs.items():
        files = glob.glob(os.path.join(folder, '*.nii*'))
        for f in files:
            subj_id = Path(f).stem.split('.')[0] # Simple ID parsing
            data.append({'subject_id': subj_id, 'path': f, 'label': label, 'type': 'subtype'})

    # Load controls (reference group, e.g., nonFTLD)
    for folder in controls_dir:
        files = glob.glob(os.path.join(folder, '*.nii*'))
        for f in files:
            subj_id = Path(f).stem.split('.')[0]
            data.append({'subject_id': subj_id, 'path': f, 'label': 'Control', 'type': 'control'})

    return pd.DataFrame(data)

#-------------------------------------------------

def parse_segstats_file(filepath):

    # Parses the 'Mean' column from mri_segstats (FreeSurfer) output files
    try:
        df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None)
        # If the first column is 'Index' and second is 'SegId', Mean is likely index 5
        mean_value = df[df[1] == 1][5].values[0]
        print(f"\nMean W-score calculated from {filepath}:")
        print(mean_value)
        return(mean_value)
            
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return 0.0 # Default if failed

####################################################################################################
# FOR TESTING
####################################################################################################

if __name__ == "__main__":
    parse_segstats_file('/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/pipeline_test/w-scores/BV/1S0207_rh_segstats.txt')