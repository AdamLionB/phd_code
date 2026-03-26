import sys
import os
import re

from phd.scripts import cupt_2_blind_cupt


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_all_blind.py <parseme_version_dir>")
        sys.exit(1)

    parseme_data_dir = sys.argv[1]

    if not os.path.isdir(parseme_data_dir):
        print(f"Error: {parseme_data_dir} is not a directory")
        sys.exit(1)
    
    sub_dirs = os.listdir(parseme_data_dir)

    if any(re.match(r'^\d+\.\d+$', subdir) for subdir in sub_dirs):
        print(f'Error: PLease provide the path to a given version, not the parent directory. Found subdirectories: {sub_dirs}')
        sys.exit(1)
    
    dir_to_ignore = {'bin', 'trial', 'system-results'}
    files_to_process = ['dev', 'test', 'train']

    for lang in sub_dirs:
        if lang in dir_to_ignore:
            print(f'Skipping {lang} as it is in the ignore list.')
            continue
        lang_dir = os.path.join(parseme_data_dir, lang)
        if not os.path.isdir(lang_dir):
            # print(f'Warning: {lang_dir} is not a directory.')
            continue
        for filename in files_to_process:
            if os.path.isfile(os.path.join(lang_dir, filename+'.blind.cupt')):
                print(f'Blind file for {lang} {filename} already exists, skipping.')
                continue
            file_path = os.path.join(lang_dir, filename+'.cupt')
            if os.path.isfile(file_path):
                print(f'Processing {file_path}')

                cupt_2_blind_cupt.main(file_path)

            else:
                print(f'Warning: {file_path} does not exist, skipping.')