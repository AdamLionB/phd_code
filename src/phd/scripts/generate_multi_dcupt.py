import sys
import os
import re
from pathlib import Path
from phd import cupt_parser, dcupt

def main(parseme_data_dir, excluded_langs: list[str] = []):
    if not os.path.isdir(parseme_data_dir):
        print(f"Error: {parseme_data_dir} is not a directory")
        sys.exit(1)
    
    sub_dirs = sorted(os.listdir(parseme_data_dir))  # sorted for reproducible base ordering

    if any(re.match(r'^\d+\.\d+$', subdir) for subdir in sub_dirs):
        print(f'Error: PLease provide the path to a given version, not the parent directory. Found subdirectories: {sub_dirs}')
        sys.exit(1)

    dir_to_ignore = {'bin', 'trial', 'system-results', 'Multi'} | set(excluded_langs)
    files_to_process = ['train', 'dev', 'test']

    if excluded_langs:
        suffix = '_no_' + '_'.join(sorted(excluded_langs))
        multi_dir = os.path.join(parseme_data_dir, f'Multi{suffix}')
    else:
        multi_dir = os.path.join(parseme_data_dir, 'Multi')
    os.makedirs(multi_dir, exist_ok=True)

    # lang -> file_type -> (blind_cupt_path, per-lang overrides keyed by (0, sent_idx))
    # Used after the main loop to build traindev interleaved by language.
    lang_file_data: dict[str, dict[str, tuple[str, dict[tuple[int, int], list[str]]]]] = {}

    for file_type in files_to_process:
        overrides: dict[tuple[int, int], list[str]] = {}
        base_refs = []

        output_path = os.path.join(multi_dir, f'{file_type}.gold.dcupt')
        output_dir = Path(output_path).resolve().parent

        for lang in sub_dirs:
            if lang in dir_to_ignore:
                continue
            lang_dir = os.path.join(parseme_data_dir, lang)
            if not os.path.isdir(lang_dir):
                continue
            
            gold_cupt_path = os.path.join(lang_dir, file_type+'.cupt')
            blind_cupt_path = os.path.join(lang_dir, file_type+'.blind.cupt')
            if not os.path.isfile(gold_cupt_path):
                print(f'Warning: {gold_cupt_path} does not exist, skipping.')
                continue
            if not os.path.isfile(blind_cupt_path):
                print(f'Warning: {blind_cupt_path} does not exist, skipping.')
                continue

            base_idx = len(base_refs)
            base_refs.append(os.path.relpath(blind_cupt_path, output_dir))

            lang_overrides: dict[tuple[int, int], list[str]] = {}
            parser = cupt_parser.Cupt_parser(gold_cupt_path)
            for sent_id, tl in enumerate(parser.read_next_n()):
                values = [token.get('parseme:mwe', '*') or '*' for token in tl]
                v = values if any(val != '*' for val in values) else []
                lang_overrides[(0, sent_id)] = v
                overrides[(base_idx, sent_id)] = v

            if file_type in ('train', 'dev'):
                lang_file_data.setdefault(lang, {})[file_type] = (blind_cupt_path, lang_overrides)

        dcupt.create(
            output_path=output_path,
            base_ref=base_refs,
            columns=['PARSEME:MWE'],
            default_value='*',
            overrides=overrides,
        )
        print(f"Generated {output_path}")
        print(f"  {len(base_refs)} bases, {sum(1 for v in overrides.values() if v)} sentences with MWE annotations")

        blind_output_path = os.path.join(multi_dir, f'{file_type}.blind.dcupt')
        dcupt.create(
            output_path=blind_output_path,
            base_ref=base_refs,
            columns=['PARSEME:MWE'],
            default_value='*',
        )
        print(f"Generated {blind_output_path}")

    # Generate Multi.traindev.gold.dcupt interleaved by language:
    # DE_train, DE_dev, EL_train, EL_dev, ...
    traindev_output_path = os.path.join(multi_dir, 'traindev.gold.dcupt')
    traindev_output_dir = Path(traindev_output_path).resolve().parent
    traindev_base_refs = []
    traindev_overrides: dict[tuple[int, int], list[str]] = {}

    for lang in sub_dirs:  # already sorted
        if lang not in lang_file_data:
            continue
        for file_type in ('train', 'dev'):
            if file_type not in lang_file_data[lang]:
                continue
            blind_path, lang_overrides = lang_file_data[lang][file_type]
            base_idx = len(traindev_base_refs)
            traindev_base_refs.append(os.path.relpath(blind_path, traindev_output_dir))
            for (_, s), v in lang_overrides.items():
                traindev_overrides[(base_idx, s)] = v

    dcupt.create(
        output_path=traindev_output_path,
        base_ref=traindev_base_refs,
        columns=['PARSEME:MWE'],
        default_value='*',
        overrides=traindev_overrides,
    )
    print(f"Generated {traindev_output_path}")
    print(f"  {len(traindev_base_refs)} bases, {sum(1 for v in traindev_overrides.values() if v)} sentences with MWE annotations")

    traindev_blind_output_path = os.path.join(multi_dir, 'traindev.blind.dcupt')
    dcupt.create(
        output_path=traindev_blind_output_path,
        base_ref=traindev_base_refs,
        columns=['PARSEME:MWE'],
        default_value='*',
    )
    print(f"Generated {traindev_blind_output_path}")

        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_multi_dcupt.py <parseme_version_dir> [LANG ...]")
        sys.exit(1)
    main(sys.argv[1], excluded_langs=sys.argv[2:])