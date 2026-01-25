
import os
import csv
import glob

# List your folders in order (first is the base, rest are merged in order)
FOLDERS = [
    '/home/asia/rok3/sad2/final_project/SAD2_final_project/data/analysis_3_synchronous_part_1',
    '/home/asia/rok3/sad2/final_project/SAD2_final_project/data/analysis_3_synchronous_part_2',
    '/home/asia/rok3/sad2/final_project/SAD2_final_project/data/analysis_3_synchronous_part_3',
    '/home/asia/rok3/sad2/final_project/SAD2_final_project/data/analysis_3_synchronous_part_4',
    '/home/asia/rok3/sad2/final_project/SAD2_final_project/data/analysis_3_synchronous_part_5',
    '/home/asia/rok3/sad2/final_project/SAD2_final_project/data/analysis_3_synchronous_part_6',
    '/home/asia/rok3/sad2/final_project/SAD2_final_project/data/analysis_3_synchronous_part_7',
]
def merge_metadata(folder1, folder2, increment):
    file1 = os.path.join(folder1, 'metadata.csv')
    file2 = os.path.join(folder2, 'metadata.csv')
    rows = []
    header = None

    # Read folder1
    if os.path.exists(file1):
        with open(file1, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.extend(list(reader))

    # Read folder2 and increment indexes
    #### do przejrzenia 
    if os.path.exists(file2):
        with open(file2, newline='') as f:
            reader = csv.reader(f)
            if header is None:
                header = next(reader)
            else:
                next(reader)  # skip header
            for row in reader:
                # Increment first column if it's a digit
                if row and row[0].isdigit():
                    row[0] = str(int(row[0]) + increment)
                # Increment zero-padded index column (column 10, index 9) if present and is digit
                if len(row) > 9 and row[9].isdigit():
                    new_index = int(row[9]) + increment
                    row[9] = str(new_index).zfill(len(row[9]))
                rows.append(row)
    # Write merged
    with open(file1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Merged metadata.csv into {file1}")


def merge_all_joined_results(folders, increments, output_file):
    all_rows = []
    header = None
    for folder, increment in zip(folders, increments):
        files = glob.glob(os.path.join(folder, 'joined_results_*.csv'))
        for file in files:
            # Extract file_id from filename (joined_results_X.csv)
            basename = os.path.basename(file)
            try:
                file_id = int(basename.split('_')[-1].split('.')[0])
            except Exception:
                print(f"Could not extract file_id from {basename}")
                continue
            k_number = file_id * 20
            # Read, add k_number column, and overwrite file
            with open(file, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                orig_header = reader.fieldnames
            # Add k_number to each row
            for row in rows:
                row['k_number'] = k_number
            # Write back with new column if not present
            if 'k_number' not in orig_header:
                new_header = orig_header + ['k_number']
            else:
                new_header = orig_header
            with open(file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=new_header)
                writer.writeheader()
                writer.writerows(rows)
            # For merging, use the updated rows
            # Adjust dataset index as before
            for row in rows:
                if 'dataset' in row and row['dataset'].isdigit():
                    row['dataset'] = str(int(row['dataset']) + increment)
                all_rows.append(row)
            # Set header for merged file
            if header is None:
                header = new_header
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Merged all joined_results_*.csv into {output_file}")

if __name__ == '__main__':
    base_folder = FOLDERS[0]
    increments = [i * 300 for i in range(len(FOLDERS))]
    # Merge metadata as before
    for i, add_folder in enumerate(FOLDERS[1:], 1):
        print(f"Merging {add_folder} into {base_folder} with increment {increments[i]}")
        merge_metadata(base_folder, add_folder, increments[i])
    # Merge all joined_results_*.csv into one file
    output_file = os.path.join(base_folder, 'joined_results_merged.csv')
    merge_all_joined_results(FOLDERS, increments, output_file)
    print("Merging of results complete.")