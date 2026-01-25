import pandas as pd
import os
import shutil
from glob import glob
# michał komp
# base_path = '/Users/chmuradin/Desktop/SAD2_final_project/data/'
# max komp
base_path = '/home/maxi7524/repositories/SAD2_final_project/data'

final_dir_name = 'analysis_synchronous'

os.chdir(base_path)

# --- CONFIGURATION ---
# LIST YOUR 4 FOLDERS HERE
standard_subfolders = ['bn_ground_truth', 'datasets', 'datasets_bnfinder']
# ---------------------
special_subfolder = 'results'
all_subfolders = [*standard_subfolders, 'results']
os.chdir(base_path)

os.chdir(base_path)


def pad_condition_id(x, offset=0):
    """
    Takes an ID, adds the cumulative offset, and returns a 5-digit string.
    """
    val = int(x) + offset
    return str(val).zfill(5)


def copy_files_for_row(row, part, final_base_path):
    """
    Physically copies and renames files for a single row of data.
    """
    # 1. Identify IDs
    original_id_raw = str(row['original_condition_id'])  # e.g. "0"
    original_id_padded = original_id_raw.zfill(5)  # e.g. "00000"
    new_id = str(row['condition_id_name'])  # e.g. "00300" (new calculated ID)

    # 2. HANDLE STANDARD FOLDERS (Exact Match or Padded Match)
    # e.g. Source: "00000.txt" -> Dest: "00300.txt"
    for sub in all_subfolders :
        source_dir = os.path.join(base_path, f'analysis_3_synchronous_part_{part}', sub)
        dest_dir = os.path.join(final_base_path, sub)

        if os.path.exists(source_dir):
            candidates = os.listdir(source_dir)
            for file in candidates:
                # Check for "0.txt", "0", "00000.txt", "00000"
                # --- CASE A: Standard "ID.ext" (e.g., "10.png" or "00010.txt") ---
                # Matches exact ID or ID + dot
                if (file == original_id_raw or
                        file == original_id_padded or
                        file.startswith(f"{original_id_raw}.") or
                        file.startswith(f"{original_id_padded}.")):

                    _, ext = os.path.splitext(file)
                    shutil.copy2(os.path.join(source_dir, file),
                                 os.path.join(dest_dir, f"{new_id}{ext}"))

                # --- CASE B: Complex "ID_suffix.ext" (e.g., "00010_XXX.sif") ---
                # Matches ID + Underscore (e.g. "10_" or "00010_")
                elif (file.startswith(f"{original_id_raw}_") or
                      file.startswith(f"{original_id_padded}_")):

                    # Determine length of the prefix to slice it off correctly
                    prefix_len = 0
                    if file.startswith(f"{original_id_padded}_"):
                        prefix_len = len(original_id_padded)
                    else:
                        prefix_len = len(original_id_raw)

                    # Keep everything after the ID (e.g. "_XXX.sif" or "_log.txt")
                    suffix = file[prefix_len:]

                    shutil.copy2(os.path.join(source_dir, file),
                                 os.path.join(dest_dir, f"{new_id}{suffix}"))

    # 3. HANDLE SPECIAL FOLDER (Pattern Match)
    # e.g. Source: "00000_Log.txt" -> Dest: "00300_Log.txt"
    if special_subfolder:
        special_source_dir = os.path.join(base_path, f'analysis_3_synchronous_part_{part}', 'results',
                                          special_subfolder)
        special_dest_dir = os.path.join(final_base_path, special_subfolder)

        if os.path.exists(special_source_dir):
            # Find all files starting with the padded original ID (e.g., "00000_")
            search_pattern = os.path.join(special_source_dir, f"{original_id_padded}_*")
            files_found = glob.glob(search_pattern)

            for file_path in files_found:
                filename = os.path.basename(file_path)

                # Remove the old ID part (length of "00000") and keep the rest
                suffix = filename[len(original_id_padded):]

                # Construct new filename
                new_filename = f"{new_id}{suffix}"

                shutil.copy2(file_path, os.path.join(special_dest_dir, new_filename))


def process_data_and_files(parts, output_metadata, output_joined):
    df_metadata_list = []
    df_joined_list = []

    # Tracks the cumulative ID count across all parts
    global_id_offset = 0

    # 1. Create Final Directory Structure
    final_base_path = os.path.join(base_path, final_dir_name)
    if not os.path.exists(final_base_path):
        os.makedirs(final_base_path)
        print(f"Created main directory: {final_base_path}")

    # Create subfolders inside final directory
    subs_to_create = standard_subfolders + ([special_subfolder] if special_subfolder else [])
    for sub in subs_to_create:
        os.makedirs(os.path.join(final_base_path, sub), exist_ok=True)

    # 2. Main Loop over Parts
    for part in parts:
        base_results_path = os.path.join(base_path, f'analysis_3_synchronous_part_{part}', 'results')
        meta_path = os.path.join(base_results_path, 'metadata.csv')
        joined_path = os.path.join(base_results_path, f'joined_results_analysis_3_synchronous_part_{part}.csv')

        if os.path.exists(meta_path):
            print(f"Processing Part {part} (ID Offset: {global_id_offset})...")

            # --- PROCESS METADATA ---
            df_meta = pd.read_csv(meta_path)

            # Store original ID temporarily to find the physical files
            df_meta['original_condition_id'] = df_meta['condition_id_name']

            # Update IDs in Metadata
            df_meta['condition_id_name'] = df_meta['condition_id_name'].apply(pad_condition_id, offset=global_id_offset)
            df_meta['condition_id_num'] = df_meta['condition_id_name'].astype(int)

            # Add tracking columns
            df_meta['k_value'] = part * 20
            df_meta['part'] = part

            # --- COPY FILES ---
            # Apply file copy logic for every row
            df_meta.apply(lambda row: copy_files_for_row(row, part, final_base_path), axis=1)

            # Cleanup and store metadata
            df_meta_clean = df_meta.drop(columns=['original_condition_id'])
            df_metadata_list.append(df_meta_clean)

            # --- PROCESS JOINED RESULTS ---
            if os.path.exists(joined_path):
                df_joined = pd.read_csv(joined_path)

                # Apply the EXACT SAME offset to joined_results
                if 'dataset' in df_joined.columns:
                    df_joined['dataset'] = df_joined['dataset'].apply(pad_condition_id,
                                                                                          offset=global_id_offset)
                    df_joined['dataset'] = df_joined['dataset'].astype(int)

                df_joined['part'] = part
                df_joined_list.append(df_joined)
            else:
                print(f"  Warning: Joined results file not found at {joined_path}")

            # --- UPDATE OFFSET FOR NEXT PART ---
            if not df_meta.empty:
                # The next part starts after the max ID of this part
                global_id_offset = int(df_meta['condition_id_num'].max()) + 1

        else:
            print(f"  Skipping Part {part}: Metadata not found.")

    # 3. Save Final CSVs
    if df_metadata_list:
        final_meta = pd.concat(df_metadata_list, ignore_index=True)
        final_meta.to_csv(os.path.join(base_path, output_metadata), index=False)
        print(f"SUCCESS: Combined Metadata saved to {output_metadata}")
        # TODO przenieść plik


    if df_joined_list:
        final_joined = pd.concat(df_joined_list, ignore_index=True)
        final_joined.to_csv(os.path.join(base_path, output_joined), index=False)
        print(f"SUCCESS: Combined Joined Results saved to {output_joined}")
        # TODO przenieść plik 


# Execute
process_data_and_files(range(1, 8), 'metadata.csv', 'results.csv')
