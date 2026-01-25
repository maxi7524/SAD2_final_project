import os
import re
import shutil

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

CHILDREN = ['results', 'datasets', 'bn_ground_truth', 'datasets_bnfinder']
FILENAME_REGEX = re.compile(r'^(\d+)(.*)$')

def count_dataset_files(folder):
    datasets_dir = os.path.join(folder, 'datasets')
    if not os.path.isdir(datasets_dir):
        return 0
    return len([f for f in os.listdir(datasets_dir) if os.path.isfile(os.path.join(datasets_dir, f))])

def merge_folder_into_base(base_folder, add_folder, children, increment):
    # base_folder: folder wejściowy
    # add_folder: folder wyjściowy gdzie wrzucamy
    for child in children:
        #-----------ustawienie ścieżek folderów
        dir1 = os.path.join(base_folder, child)
        dir2 = os.path.join(add_folder, child)

        if not os.path.isdir(dir2):
            continue
        if not os.path.isdir(dir1):
            os.makedirs(dir1)
        
        # iteracyjnie przetwarzamy pliki w tych CHILDREN
        for fname in os.listdir(dir2):
            print(f"Processing file: {fname}")
            match = FILENAME_REGEX.match(fname)

            if not match:
                continue  # skip files not starting with a number

            # nazwanie pliku
            num, rest = match.groups()
            new_num = int(num) + increment
            new_fname = f"{new_num}{rest}"
            src = os.path.join(dir2, fname)
            dst = os.path.join(dir1, new_fname)
            shutil.move(src, dst)
            print(f"Moved: {src} -> {dst}")

if __name__ == '__main__':
    base_folder = FOLDERS[0]
    increment = 0
    for add_folder in FOLDERS[1:]:
        increment = increment + 300
        print(f"Merging {add_folder} into {base_folder} with increment {increment}")
        merge_folder_into_base(base_folder, add_folder, CHILDREN, increment)
    print("Merging complete.")