####################################################################################################
# MASTER SCRIPT
# Created by: Kedar Madi
# 12/17/2025

# Calls all the functions and Python scripts necessary to:
# 1) Generate atrophy network mapping data on the fly
# 2) Use the generated data to train and test a machine learning classifier
####################################################################################################

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import paramiko
from scp import SCPClient
import matlab.engine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
from run_snpm_batch import run_snpm_analysis
from compute_w_scores_jarvis import setup_ssh_client, generate_env_setup_cmd, execute_remote_command, find_wmap_nifti_files, create_fsavg_labels, calc_subj_stats
from classifier import load_data_map, parse_segstats_file, multinomial_log_reg, support_vector_machine
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
# Configuration
####################################################################################################

##################################################
# Basic settings
##################################################

# Define master output directory (and/or create it if it doesn't exit)
MASTER_OUTPUT_DIR = '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/pipeline_test4'

# Define groups / directories (dictionary of subtypes/classes)
FTD_SUBTYPE_DIRS = {
    'BV': '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/subjects/BV',
    'PNFA': '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/subjects/PNFA',
    'SV': '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/subjects/SV',
    'PSP': '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/subjects/PSP',
    'CBS': '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/subjects/CBS'
}

# Control group (used as reference for SnPM, but not in classification)
NONFTLD_CONTROLS_DIR = ['/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/subjects/nonFTLD']

##################################################
# SnPM settings (2 group, 2 sample t-test)
##################################################

# Define (hardcoded) SnPM 'Specify' parameters (in addition to the directories specified above)
NUMBER_PERM = 10000
CLUSTER_THRESH_VAL = 0.001
EXP_MASK = '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/OLD/else_vs_nonFTLD_v2/avg152T1_gray.nii'

# Define (hardcoded) SnPM 'Inference' parameters
FWE_VAL = 0.05

##################################################
# Jarvis + FreeSurfer settings (via SSH)
##################################################

JARVIS_WORKSPACE = '/storage/home/madik/projects/ANM_NIFD_demo/temp/' # Ensure this directory exists on the remote machine (Jarvis)
JARVIS_FS_HOME = '/storage/local/freesurfer74/' # Define FreeSurfer home here
JARVIS_FSL_HOME = '/usr/local/fsl' # Define the FSL home here

# Local machine settings
LOCAL_W_MAP_PARENT_DIR = '/Users/kedarmadi/Documents/projects/ANM_NIFD_demo/w-maps/' # Local parent directory containing subject w-maps (NIfTI files)
TARGET_SUBJECT = 'fsaverage' # Assuming projection to the fsaverage space
HEMISPHERES = ['lh', 'rh']

####################################################################################################
# Create Necessary Directories
####################################################################################################

if not os.path.exists(MASTER_OUTPUT_DIR):
    os.makedirs(MASTER_OUTPUT_DIR)
    print(f"\n{MASTER_OUTPUT_DIR} + created.")

####################################################################################################
# Run Cross-Validation Classifier
####################################################################################################

##################################################
# Prepare Data Handling
##################################################

df = load_data_map(FTD_SUBTYPE_DIRS, NONFTLD_CONTROLS_DIR)

# Filter subtypes to ONLY those that have files in their respective directories
# This will exclude any subtypes that have empty directories
available_subtypes = df[df['type'] == 'subtype']['label'].unique().tolist()
print(f"Subtypes found with data: {available_subtypes}")

# Filter the dataframe to only include these subtypes in the classifier (+ controls)
classifier_df = df[df['type'].isin(['subtype', 'control'])].reset_index(drop=True)

X_paths = classifier_df['path'].values
y_labels = classifier_df['label'].values

# Optional: full list of all controls (not used directly in SnPM anymore)
all_control_paths = df[df['type'] == 'control']['path'].tolist()

# Prepare w-map files (find them once to avoid repeated scanning)
print("Indexing local w-maps...")
all_wmaps = find_wmap_nifti_files(LOCAL_W_MAP_PARENT_DIR)

# Create a lookup dictionary: subject_id -> {'lh': path, 'rh': path}
wmap_lookup = {}
for item in all_wmaps:
    sid = item['subject_id']
    if sid not in wmap_lookup:
        wmap_lookup[sid] = {}
        wmap_lookup[sid][item['hemi']] = item['path']

##################################################
# Initialize MATLAB engine
##################################################

eng = None

print("\nStarting MATLAB engine...")
print("\n...\n")

try:
    eng = matlab.engine.start_matlab()
    print("MATLAB engine started successfully.")
    print("\nContinuing...")

except Exception as e:
    print(f"Error starting MATLAB engine: {e}")
    print("Please ensure the MATLAB Engine API is correctly installed and configured.")
    sys.exit(1)

##################################################
# Establish SSH Connection to Jarvis
##################################################
        
ssh_client = None

# Load SSH config file
ssh_config = paramiko.SSHConfig()
user_config_file = os.path.expanduser("~/.ssh/config")
if os.path.exists(user_config_file):
    with open(user_config_file) as f:
        ssh_config.parse(f)

# Look up the host settings (for Jarvis)
host_config = ssh_config.lookup('jarvis')

# Get credentials for Jarvis
jarvis_host = host_config.get('hostname')
jarvis_user = host_config.get('user')

print("\nJarvis login details:")
print(f"HostName: {jarvis_host}")
print(f"User: {jarvis_user}")

print("\nTesting your connection to Jarvis...")
ssh_client = setup_ssh_client(jarvis_host, jarvis_user)

# This environment command correctly sets SUBJECTS_DIR for fsaverage processing
env_cmd = generate_env_setup_cmd(JARVIS_FSL_HOME, JARVIS_FS_HOME)
                
# Ensure the remote workspace exists
execute_remote_command(ssh_client, f"mkdir -p {JARVIS_WORKSPACE}")

##################################################
# Cross-Validation Loop
##################################################

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays to store per-fold accuracy scores
clf_fold_accuracies = []
svm_fold_accuracies = []
svm_fold_aucs = []

# Initialize array to store combined features across all folds (saved as a CSV)
all_features_across_folds = []

# Initialize array to store log. regression coefficients
coef_tables = []

# Initialize arrays to store predicted vs. true results for confusion matrix
clf_all_y_true = []
clf_all_y_pred = []
svm_all_y_true = []
svm_all_y_pred = []

try:
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_paths, y_labels)):
        print(f"\n{'#'*40} Processing FOLD {fold_idx+1} {'#'*40}")

        fold_dir = os.path.join(MASTER_OUTPUT_DIR, f"fold_{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split Train/Test
        X_train_paths, X_test_paths = X_paths[train_idx], X_paths[test_idx]
        clf_y_train, clf_y_test = y_labels[train_idx], y_labels[test_idx]

        # Build *Train controls* for this fold (no leakage)
        train_control_paths = [p for p, l in zip(X_train_paths, clf_y_train) if l == 'Control']

        # Initialize Feature Matrix for this fold (rows = all subjects, columnns = subtype masks)
        # We calculate features for BOTH train and test subjects using the TRAIN masks
        current_fold_subjects = np.concatenate([X_train_paths, X_test_paths])
        feature_matrix = pd.DataFrame(index=current_fold_subjects, columns=available_subtypes)

        with SCPClient(ssh_client.get_transport()) as scp_client:

            ##################################################
            # 1. Feature Generation (Batch SnPM Analysis -> Subtype-Specific Network Mask Creation)
            ##################################################
            
            for subtype in available_subtypes:
                print(f"\n{'='*80}\nGenerating Network Mask for {subtype} (Fold {fold_idx+1}) via SnPM Batch Analysis...\n{'='*80}")
                subtype_dir = FTD_SUBTYPE_DIRS[subtype]

                #-------------------------------------------------
                # 1A. Select ONLY Training subjects of this subtype
                
                train_subtype_paths = [p for p, l in zip(X_train_paths, clf_y_train) if l == subtype]

                if not train_subtype_paths:
                    print(f"Skipping {subtype} - no training samples.")
                    feature_matrix[subtype] = 0.0
                    continue

                #-------------------------------------------------
                # 1B. Run SnPM (Training Subtype vs. All Controls)
                
                snpm_out_dir = os.path.join(fold_dir, 'SnPM', subtype)
                os.makedirs(snpm_out_dir, exist_ok=True)
                output_stat_filename = os.path.join(snpm_out_dir, f"{subtype}_SnPM_filtered")

                # Print out parameters for verification purposes
                print(f"\nNow running the 'run_snpm_analysis()' function.")
                print(f"Analysis will be completed with the following SnPM parameters:")
                print("\nSnPM 'Specify' Parameters:")
                print(f"Group 1 directory (FTD subtype): {subtype}")
                print(f"Group 2 directory (i.e., healthy controls): {NONFTLD_CONTROLS_DIR}")
                print("Analysis (i.e., Output/Results) Directory: " + snpm_out_dir)
                print(f"Number of Permutations: {NUMBER_PERM}")
                print(f"Cluster Threshold Value: {CLUSTER_THRESH_VAL}")
                print("Explicit Mask: " + EXP_MASK)
                print("\nSnPM 'Inference' Parameters:")
                print(f"FWE value: {FWE_VAL}")
                print("Output Statistics Filename: " + output_stat_filename + ".nii")

                print("\nBeginning SnPM analysis...")

                # Call modified SnPM function (*passing lists, NOT directories*)
                run_snpm_analysis(
                    eng,
                    snpm_out_dir,
                    train_subtype_paths,
                    train_control_paths,
                    NUMBER_PERM,
                    CLUSTER_THRESH_VAL,
                    EXP_MASK, FWE_VAL,
                    output_stat_filename
                )

                mask_path = output_stat_filename + "_clean.nii"
                if not os.path.exists(mask_path):
                    print("No significant mask found. Features set to 0.")
                    feature_matrix[subtype] = 0.0
                    continue

                #-------------------------------------------------
                # 1C. Create labels on Jarvis

                print(f"\n{'='*80}\nCreating fsaverage label map for {subtype} (Fold {fold_idx+1})...\n{'='*80}")

                wscore_out_dir = os.path.join(fold_dir, 'w-scores', subtype)
                os.makedirs(wscore_out_dir, exist_ok=True)
                jarvis_mask_path = os.path.join(JARVIS_WORKSPACE, f"fold{fold_idx}_{os.path.basename(mask_path)}")

                local_fsaverage_labels_map = create_fsavg_labels(
                    ssh_client,
                    scp_client,
                    env_cmd,
                    mask_path,
                    JARVIS_WORKSPACE,
                    jarvis_mask_path,
                    TARGET_SUBJECT,
                    HEMISPHERES,
                    f"{subtype}_fold{fold_idx}",
                    wscore_out_dir
                )

                #-------------------------------------------------                
                # 1D. Compute W-scores for ALL subjects (Train + Test)

                print(f"\n{'='*80}\nComputing average W-scores for all subjects within the {subtype} (Fold {fold_idx+1}) mask...\n{'='*80}")
                
                # Resolve Subject IDs to w-map paths
                fold_wmap_files = []
                for subj_path in current_fold_subjects:
                    sid = Path(subj_path).stem.split('.')[0]
                    if sid in wmap_lookup:
                        if 'lh' in wmap_lookup[sid]: fold_wmap_files.append(wmap_lookup[sid]['lh'])
                        if 'rh' in wmap_lookup[sid]: fold_wmap_files.append(wmap_lookup[sid]['rh'])

                # Calculate stats (mri_segstats output) on Jarvis
                calc_subj_stats(
                    ssh_client,
                    scp_client,
                    local_fsaverage_labels_map,
                    env_cmd,
                    LOCAL_W_MAP_PARENT_DIR,
                    JARVIS_WORKSPACE,
                    TARGET_SUBJECT,
                    wscore_out_dir,
                    subject_file_list=fold_wmap_files
                )
                
                #-------------------------------------------------                
                # 1E. Parse stats files (mri_segstats output)

                for subj_path in current_fold_subjects:
                    base_name = Path(subj_path).stem
                    if base_name.endswith('.nii'):
                        base_name = Path(base_name).stem
                    
                    sid = base_name.split('_')[0]

                    val_lh = parse_segstats_file(os.path.join(wscore_out_dir, f"{sid}_lh_segstats.txt"))
                    val_rh = parse_segstats_file(os.path.join(wscore_out_dir, f"{sid}_rh_segstats.txt"))

                    feature_matrix.loc[subj_path, subtype] = (val_lh + val_rh) / 2.0

            # Force the entire Feature Matrix to be numeric
            # 'coerce' will turn any remaining non-numbers. into NaNs, which are then filled with 0.0 (for subjects missing w-maps or empty masks)
            feature_matrix = feature_matrix.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            
            # FOR DEBUGGING PURPOSES ONLY
            # Checks to see (per fold) if Feature Matrix is successfully getting filled with numeric data (i.e., mean W-scores per subject and per subtype)
            """print(f"\nFeature Matrix Summary for Fold {fold_idx+1}:")
            if feature_matrix.empty:
                print("!!! WARNING: Feature matrix is completely empty.")
            else:
                # Check how many actual numbers (non-NaNs) are in each column
                print("Non-NaN counts per subtype:")
                print(feature_matrix.notna().sum())
    
                # Check the first few rows to see if values are actually being stored
                print("\nFirst 5 rows of features:")
                print(feature_matrix.head())

                # Only try to describe if we have numeric data to avoid the KeyError
                if np.issubdtype(feature_matrix.values.dtype, np.number):
                    stats = feature_matrix.describe()
                    if 'mean' in stats.index:
                        print("\nMean feature values:")
                        print(stats.loc['mean'])
                    else:
                        print("\nNumeric stats calculated, but 'mean' not found in describe().")
                else:
                    print("\n!!! ERROR: Feature matrix contains non-numeric data types.")"""
            
            ##################################################
            # 2. ML Classification (Multinomial Log. Reg.)
            ##################################################

            print(f"\n{'='*80}\nTraining Classifier...\n{'='*80}")

            X_train_feat = feature_matrix.loc[X_train_paths].values
            X_test_feat = feature_matrix.loc[X_test_paths].values
            
            #-------------------------------------------------
            # Save per-fold feature matrix as a CSV

            feature_out_csv = os.path.join(fold_dir, f"features_fold_{fold_idx+1}.csv")

            # Add metadata columns for interpretability
            feature_df_out = feature_matrix.copy()
            feature_df_out["label"] = [y_labels[list(X_paths).index(p)] for p in feature_df_out.index]
            feature_df_out["fold"] = fold_idx + 1

            feature_df_out.to_csv(feature_out_csv)
            print(f"\nSaved feature matrix to: {feature_out_csv}")

            # Append to overall feature matrix (all folds)
            all_features_across_folds.append(feature_df_out)

            #-------------------------------------------------
            # Send feature data to classifier

            clf = multinomial_log_reg()
            clf.fit(X_train_feat, clf_y_train)

            clf_y_pred = clf.predict(X_test_feat)
            clf_acc = accuracy_score(clf_y_test, clf_y_pred)

            #-------------------------------------------------
            # Predicted probabilities (classifier)

            clf_y_prob = clf.predict_proba(X_test_feat)

            clf_prob_df = pd.DataFrame(clf_y_prob, columns=clf.named_steps["clf"].classes_)

            clf_prob_df["true_label"] = clf_y_test
            clf_prob_df["predicted_label"] = clf_y_pred
            clf_prob_df["fold"] = fold_idx + 1
            clf_prob_df["subject_path"] = X_test_paths

            clf_prob_csv = os.path.join(fold_dir, f"predicted_probabilities_fold_{fold_idx+1}.csv")
            clf_prob_df.to_csv(clf_prob_csv, index=False)
            print(f"\nSaved predicted probabilities to: {clf_prob_csv}")

            #-------------------------------------------------
            # Extract per-fold log. reg. coefficients, save as CSV (classifier)

            coef = clf.named_steps["clf"].coef_
            classes = list(clf.named_steps["clf"].classes_)

            coef_df = pd.DataFrame(coef, columns=available_subtypes, index=classes)

            coef_df["fold"] = fold_idx + 1

            coef_tables.append(coef_df)

            coef_csv = os.path.join(fold_dir, f"logreg_coefficients_fold_{fold_idx+1}.csv")
            coef_df.to_csv(coef_csv)
            print(f"\nSaved coefficients to: {coef_csv}")

            #-------------------------------------------------
            # Permutation importance (test set) (classifier)

            perm = permutation_importance(
                clf,
                X_test_feat,
                clf_y_test,
                n_repeats=50,
                random_state=42,
                scoring="accuracy"
            )

            perm_df = pd.DataFrame({
                "feature": available_subtypes,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std
            })
            
            perm_df["fold"] = fold_idx + 1

            perm_csv = os.path.join(fold_dir, f"permutation_importance_fold_{fold_idx+1}.csv")
            perm_df.to_csv(perm_csv, index=False)
            print(f"\nSaved permutation importance to: {perm_csv}")

            #-------------------------------------------------
            # Store results for the confusion matrix (classifier)

            clf_all_y_true.extend(clf_y_test)
            clf_all_y_pred.extend(clf_y_pred)

            #-------------------------------------------------
            # Print per-fold metrics (classifier)
            
            print(f"\nFold {fold_idx+1} Accuracy: {clf_acc:.4f}")
            print(f"\n")
            print(classification_report(clf_y_test, clf_y_pred, zero_division=0))
            clf_fold_accuracies.append(clf_acc)

            ##################################################
            # 3. ML Classification (Binary SVM) 
            ##################################################

            print(f"\n{'='*80}\nTraining SVM...\n{'='*80}")
            
            #-------------------------------------------------
            # Build binary labels (FTD = 1) for any subtype; Control = 0

            def to_binary(labels):
                return np.array([0 if lbl == 'Control' else 1 for lbl in labels], dtype=int)

            svm_y_train = to_binary(clf_y_train)
            svm_y_test = to_binary(clf_y_test)

            if len(np.unique(svm_y_train)) < 2: # Guard: if a fold has only one class in training, skip SVM
                print("Skipping SVM in this fold (training set has only one class).")
            
            else:

                #-------------------------------------------------
                # Send (binarized) feature data to SVM
                
                svm = support_vector_machine()
                svm.fit(X_train_feat, svm_y_train)

                svm_y_pred = svm.predict(X_test_feat)
                svm_acc = accuracy_score(svm_y_test, svm_y_pred)
                svm_fold_accuracies.append(svm_acc)

                #-------------------------------------------------
                # AUC (needs probability estimates) (SVM)

                try:
                    svm_y_prob = svm.predict_proba(X_test_feat) # shape: (n_test, 2)
                    
                    # By convention, classes_ are [0,1] = [Control, FTD]
                    svm_classes = svm.named_steps["svm"].classes_.tolist()

                    # Find column index for positive class (FTD = 1)
                    pos_idx = svm_classes.index(1)
                    svm_auc = roc_auc_score(svm_y_test, svm_y_prob[:, pos_idx])
                    svm_fold_aucs.append(svm_auc)

                except Exception as e:
                    print(f"Could not compute probabilities/AUC: {e}")
                    svm_y_prob = None

                #-------------------------------------------------
                # Store for global confusion matrix (SVM)

                svm_all_y_true.extend(svm_y_test)
                svm_all_y_pred.extend(svm_y_pred)

                print(f"\nBinary SVM - Fold {fold_idx+1} Accuracy: {svm_acc:.4f}")

                if svm_y_prob is not None and len(svm_fold_aucs) == len (svm_fold_accuracies):
                    print(f"Binary SVM - Fold {fold_idx+1} ROC AUC: {svm_auc:.4f}")

                #-------------------------------------------------
                # Save per-fold outputs as CSV (SVM)

                # Predicted probabilities
                if svm_y_prob is not None:
                    svm_prob_df = pd.DataFrame(svm_y_prob, columns=["Control_prob", "FTD_prob"]) # class order: [0, 1]
                    svm_prob_df["svm_true_label"] = svm_y_test
                    svm_prob_df["svm_predicted_label"] = svm_y_pred
                    svm_prob_df["fold"] = fold_idx + 1
                    svm_prob_df["subject_path"] = X_test_paths
                    
                    svm_prob_csv = os.path.join(fold_dir, f"svm_probabilities_fold_{fold_idx+1}.csv")
                    svm_prob_df.to_csv(svm_prob_csv, index=False)
                    print(f"Saved SVM predicted probabilities (binary classification: FTD vs. Controls) to: {svm_prob_csv}")

                # Classification report
                print("\nSVM Classification Report (FTD = 1 vs. Control = 0):\n")
                print(classification_report(svm_y_test, svm_y_pred, zero_division=0, target_names=["Control","FTD"]))

finally:
    if ssh_client: ssh_client.close() # Close SSH connection to Jarvis
    if eng: eng.quit()

####################################################################################################
# Save/Print Final Results for SVM + Classifier
####################################################################################################

#-------------------------------------------------
# Overall accuracy + classification report

# SVM
print(f"\n{'#'*30} Overall SVM Results (FTD vs. Controls) {'#'*30}")
if len(svm_fold_accuracies) > 0:
    print(f"\nAverage Accuracy (SVM): {np.mean(svm_fold_accuracies):.4f}")
    if len(svm_fold_aucs) > 0:
        print(f"Average ROC AUC (SVM): {np.mean(svm_fold_aucs):.4f}")
    
    else:
        print("No valid folds for SVM (training set had only one class in some folds).")

# Classifier (multinomial log. reg.)
print(f"\n{'#'*30} Overall Classifier Results (All FTD Subtypes + Controls) {'#'*30}")
print(f"\nAverage Accuracy: {np.mean(clf_fold_accuracies):.4f}")
print("\nClassification Report:\n")
print(classification_report(clf_all_y_true, clf_all_y_pred))

#-------------------------------------------------
# Save combined feature matrix (all folds) as CSV

features_all = pd.concat(all_features_across_folds).reset_index()
features_all.rename(columns={"index": "subject_path"}, inplace=True)

features_all_csv = os.path.join(MASTER_OUTPUT_DIR, "features_all_folds.csv")

features_all.to_csv(features_all_csv, index=False)

print(f"Saved ALL features to: {features_all_csv}")

#-------------------------------------------------
# Save CSV of average coefficients (classifier) across folds

coef_all = pd.concat(coef_tables)
coef_summary = coef_all.groupby(level=0).mean()

coef_summary_csv = os.path.join(MASTER_OUTPUT_DIR, "logreg_coefficients_mean.csv")
coef_summary.to_csv(coef_summary_csv)
print(f"Saved mean coefficients to: {coef_summary_csv}")

#-------------------------------------------------
# Generate confusion matrices

# SVM
if len(svm_all_y_true) > 0:
    svm_cm = confusion_matrix(svm_all_y_true, svm_all_y_pred, labels=[0,1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Purples', xticklabels=["Control","FTD"], yticklabels=["Control","FTD"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SVM Confusion Matrix (Control vs. FTD)")
    plt.tight_layout()
    plt.show()

# Classifier
available_classes = sorted(list(set(y_labels))) # e.g., ['BV', 'CBS', 'Control', 'PNFA', 'PSP', 'SV']

clf_cm = confusion_matrix(clf_all_y_true, clf_all_y_pred, labels = available_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(clf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=available_classes, yticklabels=available_classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (All FTD Subtypes + Controls)")
plt.show()

####################################################################################################
# End of script
####################################################################################################

print("\nAll finished!\n")