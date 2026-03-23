# %%
## Script to convert the cupt file to a blind cupt file

# takes a file as arg

import sys
import os
import re




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cupt_2_blind_cupt.py <cupt_file>")
        sys.exit(1)

    cupt_file = sys.argv[1]

    if not os.path.isfile(cupt_file):
        print(f"Error: {cupt_file} is not a file")
        sys.exit(1)

    with open(cupt_file, "r", encoding="utf-8") as in_f:
        blind_cupt_file = cupt_file.replace(".cupt", ".blind.cupt")
        with open(blind_cupt_file, "w", encoding="utf-8") as out_f:
            # read first header, (start with `# global.columns =`)
            header = in_f.readline()
            
            # find which column contains PARSEME:MWE (each column is separated by a space)
            columns = header.strip().replace("# global.columns = ", "").split(" ")
            mwe_col = columns.index("PARSEME:MWE")
            print(f"Found PARSEME:MWE in column {mwe_col}")

            # read each line
            # if line is not a comment line (starts with #) and not empty, replace the mwe_col column (the label) with `_`
            # write the line to a new file with the same name but with .blind.cupt extension
            
            out_f.write(header + "\n")
            
            for line in in_f:
                if line.startswith("#") or line.strip() == "":
                    out_f.write(line)
                    pass
                else:
                    parts = line.strip().split("\t")
                    parts[mwe_col] = "_"
                    out_f.write("\t".join(parts) + "\n")
            print(f"Blind cupt file written to {blind_cupt_file}")