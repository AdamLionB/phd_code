import sys
import os
from pathlib import Path
from phd import cupt_parser, dcupt

def main(blind_cupt_file, cupt_file, output_path):
    if not os.path.isfile(blind_cupt_file):
        print(f"Error: {blind_cupt_file} is not a file")
        sys.exit(1)

    if not os.path.isfile(cupt_file):
        print(f"Error: {cupt_file} is not a file")
        sys.exit(1)

    parser = cupt_parser.Cupt_parser(cupt_file)

    # Extract parseme:mwe per sentence.
    # Every sentence gets an entry: annotated ones carry per-token values,
    # unannotated ones use an empty list (default_value='*' covers them).
    # Having an entry for every sentence also means the dcupt file lists all
    # sentences explicitly, so the full base is included in the output.
    overrides: dict[tuple[int, int], list[str]] = {}
    for sent_id, tl in enumerate(parser.read_next_n()):
        values = [token.get('parseme:mwe', '*') or '*' for token in tl]
        if any(v != '*' for v in values):
            overrides[(0, sent_id)] = values
        else:
            overrides[(0, sent_id)] = []  # include sentence; default_value fills all tokens

    
    # output_path = output_path
    if output_path.endswith('/'):
        output_path += cupt_file.rsplit('/', 1)[-1].removesuffix('.cupt') + '.dcupt'
    if not output_path.endswith('.dcupt'):
        output_path += '.dcupt'
    output_dir = Path(output_path).resolve().parent
    base_ref = os.path.relpath(blind_cupt_file, output_dir)

    dcupt.create(
        output_path=output_path,
        base_ref=base_ref,
        columns=['PARSEME:MWE'],
        default_value='*',
        overrides=overrides,
    )

    print(f"Generated {output_path}")
    print(f"  base: {base_ref}")
    print(f"  {len(overrides)} sentences with MWE annotations")
    
    


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_gold_dcupt.py <blind.cupt file> <base .cupt file> <output .dcupt path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

