import paramiko
import os
import sys
import glob
import getpass
from pathlib import Path

####################################################################################################
# Functions
####################################################################################################

##################################################
# Helper Functions
##################################################

def setup_ssh_client(host, user):
    
    # Prompts for a password and retries until a successful SSH connection is made.
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    while True:
        
        # Prompt for password inside the loop so it can be re-entered
        jarvis_pw = getpass.getpass(f"\nTo continue, please enter your SSH password for {user}@{host}: ")
        
        try:
            print(f"\nAttempting to connect to {host}...")
            print("...")
            ssh_client.connect(
                hostname=host,
                username=user,
                password=jarvis_pw,
                timeout=10
            )
            print("✓ Connection successful.")
            return ssh_client  # Exit function with the connected client
            
        except paramiko.AuthenticationException:
            print("✗ Authentication failed. Please try again.")
            # The loop continues, prompting for the password again
            
        except paramiko.SSHException as e:
            print(f"✗ SSH error occurred: {e}")
            sys.exit(1) # General SSH errors usually aren't fixed by re-typing the password
            
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            sys.exit(1)

#-------------------------------------------------

def execute_remote_command(ssh_client, command):
    
    # Executes a command on the remote server, prints output, and blocks until it finishes

    stdin, stdout, stderr = ssh_client.exec_command(command)

    # Wait for the command to finish and get exit status
    exit_status = stdout.channel.recv_exit_status()

    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    
    if exit_status != 0:
        print(f"COMMAND FAILED (Exit {exit_status})")
        print(f"STDERR: {err}")

    elif out:
        print(f"STDOUT: {out}")

    return exit_status

#-------------------------------------------------

def transfer_files_scp(scp_client, local_path, remote_path, direction='put', recursive=False):

    # Transfers files/directories using SCPClient

    try:
        if direction == 'put':
            print(f"Transferring local '{local_path}' to remote '{remote_path}' (recursive={recursive})")
            scp_client.put(local_path, remote_path, recursive=recursive)
            print("Transfer complete.")
        
        elif direction == 'get':
            print(f"Transferring remote '{remote_path}' to local '{local_path}' (recursive={recursive})")
            scp_client.get(remote_path, local_path, recursive=recursive)
            print("Transfer complete.")
    
    except Exception as e:
        print(f"File transfer failed: {e}")

#-------------------------------------------------

def generate_env_setup_cmd(fsl_home, fs_home):

    # Generates the necessary shell commands to set up the FSL and FreeSurfer environments

    return (
        f"export FSLDIR={fsl_home} && source $FSLDIR/etc/fslconf/fsl.sh && "
        f"export FREESURFER_HOME={fs_home} && source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        f"export SUBJECTS_DIR={fs_home}/subjects &&"
    )

#-------------------------------------------------

def find_wmap_nifti_files(parent_dir):
    
    # Finds all .nii or .nii.gz files within the parent directory and its subdirectories.
    # Returns a list of dictionaries with 'path', 'subject_id', and 'hemi'.
    # Assumes filenames contain 'lh' or 'rh' and a unique subject identifier.

    print(f"Searching for w-map NIfTI files recursively within: {parent_dir}")
    wmap_files_data = []
    
    # Use glob.glob with recursive=True to find all matching files
    search_pattern = os.path.join(parent_dir, '**', '*.nii*')
    # The 'recursive=True' flag makes the '**' wildcard search subdirectories
    file_paths = glob.glob(search_pattern, recursive=True)

    for file_path in file_paths:
        # Check that the path is actually a file (glob recursive search can sometimes pick up dirs depending on pattern)
        if os.path.isfile(file_path):
            f = os.path.basename(file_path)
            
            # Remove extension(s) for parsing:
            base_name = Path(f).stem
            
            if base_name.endswith('.nii'): # Handle double extensions
                 base_name = Path(base_name).stem

            # Parse subject_id and hemi from filename
            hemi = None
            subject_id = None
            
            if 'lh' in base_name:
                hemi = 'lh'
                # Simplified parsing: replace hemi string and take everything before the first period
                subject_id = base_name.replace('_result', '').replace('_lh', '').replace('lh.', '').split('.')[0]
                # print(f"LH subject: {subject_id}") # Optional: uncomment for debugging
            
            elif 'rh' in base_name:
                hemi = 'rh'
                # Simplified parsing
                subject_id = base_name.replace('_result', '').replace('_rh', '').replace('rh.', '').split('.')[0]
                # print(f"RH subject: {subject_id}") # Optional: uncomment for debugging
            
            if subject_id and hemi:
                wmap_files_data.append({
                    'path': file_path,
                    'subject_id': subject_id,
                    'hemi': hemi
                })
            
            else:
                print(f"Warning: Could not parse subject_id/hemi from filename: {f}")
            
    print(f"Found {len(wmap_files_data)} NIfTI files with identifiable hemispheres.")
    return wmap_files_data

##################################################
# Main calculation funtions
##################################################

def create_fsavg_labels(ssh_client, scp_client, env_cmd, local_mask_path, remote_workspace, remote_mask_path, target_subject, hemispheres, output_label_suffix, local_output_dir):

    # Transforms SnPM's output network mask into an fsaverage-space label by:
    # 1. Binarizing (and uncompressesing) the network mask
    # 2. Creating a label in fsaverage space via FS's mri_vol2surf and mri_vol2label
    
    remote_binarized_mask_base = remote_mask_path.replace('.nii', '_bin.nii').replace('.gz', '_bin.nii.gz')
    remote_binarized_mask_unzipped = remote_binarized_mask_base.replace('.gz', '')

    remote_files_to_cleanup = [remote_mask_path, remote_binarized_mask_unzipped, remote_binarized_mask_base]
    remote_output_files = [] 

    # 1. Transfer input mask (SnPM's output network mask)
    transfer_files_scp(scp_client, local_mask_path, remote_mask_path, direction='put')

    # 2. Binarize (fslmaths -bin) and uncompress (gunzip) the network mask
    cmd_fslmaths = f"{env_cmd} fslmaths {remote_mask_path} -bin {remote_binarized_mask_base} -odt float"
    execute_remote_command(ssh_client, cmd_fslmaths)
    cmd_gunzip = f"{env_cmd} gunzip {remote_binarized_mask_base}"
    execute_remote_command(ssh_client, cmd_gunzip)

    # 2C + 2D: Run mri_vol2surf and mri_vol2label on fsaverage-space label mask
    print("\nCreating fsaverage-space labels. Executing FreeSurfer mri_vol2surf and mri_vol2label commands...")
    
    for hemi in hemispheres:
        remote_surf_output_mgh = os.path.join(remote_workspace, f"{hemi}.{output_label_suffix}.mgh")
        remote_label_output = os.path.join(remote_workspace, f"{hemi}.{output_label_suffix}.label")
        
        remote_files_to_cleanup.extend([remote_surf_output_mgh, remote_label_output])
        remote_output_files.extend([remote_surf_output_mgh, remote_label_output])

        cmd_vol2surf = f"{env_cmd} mri_vol2surf --mov {remote_binarized_mask_unzipped} --reg $FREESURFER_HOME/average/mni152.register.dat --hemi {hemi} --interp nearest --projfrac 0.5 --out {remote_surf_output_mgh} --trgsubject {target_subject}"
        execute_remote_command(ssh_client, cmd_vol2surf)

        cmd_vol2label = f"{env_cmd} mri_vol2label --i {remote_surf_output_mgh} --surf {target_subject} {hemi} --id 1 --l {remote_label_output}"
        execute_remote_command(ssh_client, cmd_vol2label)
    
    print("\n--- Label Generation Finished. ---")

    # 3. Transfer results back locally (only labels needed for next step)
    local_label_paths_map = {}
    
    for remote_file in remote_output_files:
        local_output_path = os.path.join(local_output_dir, os.path.basename(remote_file))
        transfer_files_scp(scp_client, local_output_path, remote_file, direction='get')
        
        if remote_file.endswith('.label'):
            # Store paths in a dictionary mapped by hemisphere for easy lookup
            hemi = Path(local_output_path).stem.split('.')[0]
            local_label_paths_map[hemi] = local_output_path
            
    print("Label results retrieved locally.")

    # 4. Clean up temporary files on the remote server
    cleanup_cmd = f"rm -f {' '.join(remote_files_to_cleanup)}"
    execute_remote_command(ssh_client, cleanup_cmd)
    print("Remote workspace cleanup complete.")

    return local_label_paths_map

#-------------------------------------------------

def calc_subj_stats(ssh_client, scp_client, local_fsaverage_labels_map, env_cmd, local_w_map_parent_dir, remote_workspace, target_subject, local_output_dir, subject_file_list=None):

    # Iterates through all subject w-map NIfTI files and uses mri_segstats
    # with the *hemisphere-matched* fsaverage-space label mask to compute statistics remotely.

    print("\n--- Starting subject-specific processing (mri_segstats) ---")

    if subject_file_list: # If a specific list of files is provided, parse it
        
        local_wmap_files_data = []
        
        for file_path in subject_file_list:
            f = os.path.basename(file_path)
            base_name = Path(f).stem
            
            if base_name.endswith('.nii'): base_name = Path(base_name).stem

            hemi = 'lh' if 'lh' in base_name else 'rh' if 'rh' in base_name else None
            subject_id = base_name.replace('_result', '').replace('_lh', '').replace('lh.', '').replace('_rh', '').replace('rh.', '').split('.')[0]

            if hemi and subject_id:
                local_wmap_files_data.append({'path': file_path, 'subject_id': subject_id,  'hemi': hemi})

    else: # Default behavior: Search the directory and find all local subject NIfTI w-maps
        local_wmap_files_data = find_wmap_nifti_files(local_w_map_parent_dir)

    # 1. Upload fsaverage labels to remote workspace once
    remote_fsaverage_labels_map = {}
    
    for hemi, local_path in local_fsaverage_labels_map.items():
        remote_path = os.path.join(remote_workspace, os.path.basename(local_path))
        transfer_files_scp(scp_client, local_path, remote_path, direction='put')
        remote_fsaverage_labels_map[hemi] = remote_path 
    
    # 2. Process each subject w-map file
    for subject_data in local_wmap_files_data:
        local_wmap_path = subject_data['path']
        subject_id = subject_data['subject_id']
        hemi = subject_data['hemi']
        filename = os.path.basename(local_wmap_path)
        remote_wmap_path = os.path.join(remote_workspace, filename)
        
        print(f"\nProcessing w-map for subject ID: {subject_id}, Hemisphere: {hemi}")

        # Transfer the individual subject w-map to the remote workspace
        transfer_files_scp(scp_client, local_wmap_path, remote_wmap_path, direction='put')

        # Get the correct remote label path for the current hemisphere
        remote_label_path = remote_fsaverage_labels_map[hemi]
        remote_stats_output = os.path.join(remote_workspace, f"{subject_id}_{hemi}_segstats.txt")
        local_stats_output = os.path.join(local_output_dir, f"{subject_id}_{hemi}_segstats.txt")

        # mri_segstats command
        cmd_segstats = (
            f"{env_cmd} mri_segstats "
            f"--slabel {target_subject} {hemi} {remote_label_path} " # Correct syntax for labels
            f"--in {remote_wmap_path} "
            f"--avgwf --sum "
            f"--nonempty "
            f"--o {remote_stats_output}"
        )

        # Note: If your w-maps are *not* already registered/aligned to fsaverage space/MNI space, 
        # mri_segstats might produce incorrect results.

        # Run the command and check if it actually created the file
        status = execute_remote_command(ssh_client, cmd_segstats)

        if status == 0:
            # Transfer the final stats file back to the local machine, but only if the command succeeded
            transfer_files_scp(scp_client, local_stats_output, remote_stats_output, direction='get')

        else:
            print(f"Skipping transfer for {subject_id} {hemi} due to remote error.")    
        
        # Clean up remote files for this iteration
        execute_remote_command(ssh_client, f"rm -f {remote_stats_output} {remote_wmap_path}")

    # Clean up the remote fsaverage label files after all subjects are processed
    cleanup_labels_cmd = f"rm -f {' '.join(remote_fsaverage_labels_map.values())}"
    execute_remote_command(ssh_client, cleanup_labels_cmd)
        
    print("\n--- Subject-specific processing complete for all subjects. ---")