import matlab.engine
from fsl.wrappers import fslmaths
import os
import glob
import sys

####################################################################################################
# Functions
####################################################################################################

##################################################
# Helper Functions
##################################################

def print_file_summary(file_list, group_name):
    
    # Prints the first 3 and last 3 items of a file list, with an ellipsis in between if the list is long enough.
    
    num_files = len(file_list)
    print(f"\nFound {num_files} NIfTI files for {group_name}")
    
    if num_files > 6:
        for f in file_list[:3]:
            print(f)
        print("...")
        for f in file_list[-3:]:
            print(f)
    else:
        # If 6 or fewer files, just print them all
        for f in file_list:
            print(f)

#-------------------------------------------------

def check_and_list_nii_files(directories):
    
    # Checks directories for .nii or .nii.gz files and returns full paths.
    
    files_list = []

    for directory in directories:
        
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
        
        # Search for .nii and .nii.gz files
        nii_files = glob.glob(os.path.join(directory, '*.nii'))
        nii_gz_files = glob.glob(os.path.join(directory, '*.nii.gz'))
        found_files = nii_files + nii_gz_files
        
        if not found_files:
            print(f"Warning: No .nii files found in directory: {directory}")
        else:
            # Append ',1' to each file path as required by SPM/SnPM
            spm_formatted_files = [f"{file},1" for file in found_files]
            files_list.extend(spm_formatted_files)

    return files_list

##################################################
# Main calculation funtions
##################################################

def run_snpm_analysis(mtlb_eng, output_dir, group1_input, group2_input, number_perm, cluster_thresh_val, exp_mask, FWE_val, output_stat_filename):
    
    # Runs an SnPM two-sample t-test analysis using the MATLAB engine.

    # Args:
        # output_dir: Path of directory where SnPM's output/results will be placed.
        # group1_input (list): List of paths to files OR directories containing Group 1 scans.
        # group2_input (list): List of paths to files OR directories containing Group 2 scans.
        # number_perm: Number of permutations (default: 5000)
        # cluster_thresh_val: Cluster thresholding value (default: 3.09)
        # exp_mask: Path to explicit mask, if wanted.
        # FWE_val: Value for family-wise error rate correction
        # output_stat_filename: Desired name (basename) for the statistical map (.nii file) showing the results. Do not include full path or extension.

    '''##################################################
    # 1. MATLAB engine check
    ##################################################
    
    print("\n--------------------------------------------------")
    print("\nStarting MATLAB engine...")
    print("\n...\n")

    try:
        eng = matlab.engine.start_matlab()
        print("MATLAB engine started successfully.")

        print("\n--------------------------------------------------")
        print("\nBeginning SnPM groupwise analysis...")
    
    except Exception as e:
        print(f"Error starting MATLAB engine: {e}")
        print("Please ensure the MATLAB Engine API is correctly installed and configured.")
        sys.exit(1)'''

    ##################################################
    # 2. Check inputs and list NIfTI files
    ##################################################

    # Helper to determine if input is a DIRECTORY list or FILE list
    def get_files(input_data):
        
        # If the first item is a directory, assume it's a list of directories
        if os.path.isdir(input_data[0]):
            return check_and_list_nii_files(input_data)
        
        else:
            # Assume it is already a list of file paths.
            # We must still append ',1' for SPM/SnPM if not present
            return [f if f.endswith(',1') else f"{f},1" for f in input_data]
        
    ##################################################

    group1_files = get_files(group1_input)
    group2_files = get_files(group2_input)

    if not group1_files or not group2_files:
        print("Error: Either Group 1 or Group 2 has no valid .nii files. Aborting...")
        mtlb_eng.quit()
        sys.exit()

    # Print files in each group for verification purposes
    print_file_summary(group1_files, "Group 1 (Training Set):")
    print_file_summary(group2_files, "Group 2 (Controls):")

    ##################################################
    # 3. Call the MATLAB function to run the batch
    ##################################################

    try:
        # Pass the cell arrays to the MATLAB function
        mtlb_eng.run_snpm_2G2STT_batch(output_dir, group1_files, group2_files, number_perm, cluster_thresh_val, exp_mask, FWE_val, output_stat_filename, nargout=0)
        print("\nSnPM analysis finished.")
        fsl_remove_nans(output_stat_filename)
    
    except matlab.engine.MatlabExecutionError as e:
        print(f"Error during MATLAB execution: {e}")

#-------------------------------------------------

def fsl_remove_nans(output_stat_filename):
    
    # Runs an fslmaths command to remove the NaNs from output image that SnPM generates.

    os.environ['FSLOUTPUTTYPE'] = 'NIFTI' # Set FSL to output uncompressed .nii files
    
    input_file_path = output_stat_filename + ".nii"
    clean_output_file_path = output_stat_filename + "_clean.nii"

    print("\n--------------------------------------------------")
    print(f"\nRemoving NaNs from output image: {input_file_path}")
    print("\n...\n")

    try:
        
        fslmaths(input_file_path).nan().run(clean_output_file_path)
        print(f"FSL command executed successfully, output saved to {clean_output_file_path}\n")
    
    except Exception as e:
        
        print(f"An error occurred: {e}")