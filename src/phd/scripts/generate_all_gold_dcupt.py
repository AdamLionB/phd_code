import sys
import os
import re
from pathlib import Path

from phd.scripts import generate_gold_dcupt
from phd.dcupt import create, read_overrides

def main(parseme_data_dir):
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
            gold_cupt_path = os.path.join(lang_dir, filename+'.cupt')
            blind_cupt_path = os.path.join(lang_dir, filename+'.blind.cupt')
            gold_dcupt_path = os.path.join(lang_dir, filename+'.gold.dcupt')
            
            if os.path.isfile(gold_dcupt_path):
                print(f'Gold dcupt file for {lang} {filename} already exists, skipping.')
                continue
            if not os.path.isfile(gold_cupt_path):
                print(f'Warning: {gold_cupt_path} does not exist, skipping.')
                continue
            if not os.path.isfile(blind_cupt_path):
                print(f'Warning: {blind_cupt_path} does not exist, skipping.')
                continue
            
            generate_gold_dcupt.main(blind_cupt_path, gold_cupt_path)
        
        traindev_dcupt_path = os.path.join(lang_dir, 'traindev.gold.dcupt')
        if os.path.isfile(traindev_dcupt_path):
            print(f'Gold traindev dcupt file for {lang} already exists, skipping.')
            continue

        train_dcupt_path = os.path.join(lang_dir, 'train.gold.dcupt')
        dev_dcupt_path = os.path.join(lang_dir, 'dev.gold.dcupt')
        if not (os.path.isfile(train_dcupt_path) and os.path.isfile(dev_dcupt_path)):
            print(f'Warning: Missing train or dev gold dcupt for {lang}, skipping traindev gold dcupt generation.')
            continue
        
        # generate traindev.gold.dcupt with two bases (train.blind.cupt + dev.blind.cupt)
        # Sentence indices are (base_index, sent_index): train → base 0, dev → base 1.
        train_blind_path = os.path.join(lang_dir, 'train.blind.cupt')
        dev_blind_path = os.path.join(lang_dir, 'dev.blind.cupt')

        train_overrides = read_overrides(train_dcupt_path)  # keys: (0, k)
        dev_overrides = {(1, k): v for (_, k), v in read_overrides(dev_dcupt_path).items()}

        merged_overrides = {**train_overrides, **dev_overrides}

        output_dir = Path(traindev_dcupt_path).resolve().parent
        base_refs = [
            os.path.relpath(train_blind_path, output_dir),
            os.path.relpath(dev_blind_path, output_dir),
        ]

        create(
            output_path=traindev_dcupt_path,
            base_ref=base_refs,
            columns=['PARSEME:MWE'],
            default_value='*',
            overrides=merged_overrides,
        )
        print(f"Generated {traindev_dcupt_path}")

        # generate traindev.blind.dcupt by merging train.blind.cupt and dev.blind.cupt.
        # All PARSEME:MWE values are already '*' in blind files, so default_value covers
        # everything — no per-sentence overrides needed.
        traindev_blind_path = os.path.join(lang_dir, 'traindev.blind.dcupt')

        create(
            output_path=traindev_blind_path,
            base_ref=base_refs,
            columns=['PARSEME:MWE'],
            default_value='*',
        )
        print(f"Generated {traindev_blind_path}")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_all_gold_dcupt.py <parseme_version_dir>")
        sys.exit(1)
    main(sys.argv[1])