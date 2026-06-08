"""
Run the MTLB training + prediction + dcupt-generation pipeline for one or more
languages, repeated 3 times each (incrementing a counter per output file).

Layout assumptions
------------------
Workspace root   : /users/21504193t/
MTLB code dir    : <root>/MTLB-STRUCT/code/          (has its own uv venv)
phd_code dir     : <root>/phd_code/                   (has its own uv venv)
PARSEME blind    : <phd_code>/src/phd/data/parseme/1.2/{lang}/test.blind.cupt
dcupt output     : <phd_code>/src/phd/data/mtlb_pred/1.2/{lang}/
generate script  : <phd_code>/src/phd/scripts/generate_dcupt.py

Usage
-----
    python scripts/run_mtlb_pipeline.py DE
    python scripts/run_mtlb_pipeline.py DE FR HE
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# File lives at phd_code/src/phd/scripts/run_mtlb_pipeline.py
PHD_CODE   = Path(__file__).resolve().parents[3]          # phd_code/
WORKSPACE  = PHD_CODE.parent                              # /users/21504193t/
MTLB_CODE  = WORKSPACE / "MTLB-STRUCT" / "code"

PARSEME_DATA    = PHD_CODE / "src" / "phd" / "data" / "parseme" / "1.2"
MTLB_PRED_BASE  = PHD_CODE / "src" / "phd" / "data" / "mtlb_pred" / "1.2"
GENERATE_SCRIPT = Path(__file__).resolve().parent / "generate_dcupt.py"

NUM_RUNS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess, streaming output, raising on non-zero exit."""
    print(f"\n[CMD] {' '.join(str(c) for c in cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_config(path: Path, cfg: dict) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def saved_folder_name(cfg: dict, mode: str) -> Path:
    """
    Reproduce the folder name that main.py builds:
        {save_dir}{LANG}_{mode}_{PRETRAIN_MODEL.replace('/', '-')}_{single|multitask}
    Returns an *absolute* path.
    """
    lang    = cfg["data"]["language"]
    model   = cfg["model"]["pretrained_model_name"].replace("/", "-")
    task    = "multitask" if cfg["model"]["multi_task"] else "single"
    save_dir = MTLB_CODE / cfg["training"]["save_dir"]   # save_dir may be relative like ./saved/
    return (save_dir / f"{lang}_{mode}_{model}_{task}").resolve()


def next_counter(lang: str, suffix: str) -> int:
    """
    Find the next integer counter for dcupt files of the form
        test.mtlb_trained_on_{suffix}_{N}.dcupt
    in the output directory for <lang>.
    """
    out_dir = MTLB_PRED_BASE / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"test\.mtlb_trained_on_{re.escape(suffix)}_(\d+)\.dcupt")
    existing = [int(m.group(1)) for f in out_dir.iterdir()
                if (m := pattern.match(f.name))]
    return max(existing, default=-1) + 1


def generate_dcupt(lang: str, prediction_cupt: Path, suffix: str, counter: int) -> None:
    """Call generate_dcupt.py via phd_code's uv venv."""
    blind_cupt = PARSEME_DATA / lang / "test.blind.cupt"
    if not blind_cupt.exists():
        sys.exit(f"Blind cupt not found: {blind_cupt}")

    out_dir = MTLB_PRED_BASE / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"test.mtlb_trained_on_{suffix}_{counter}.dcupt"

    run(
        ["uv", "run", str(GENERATE_SCRIPT),
         str(blind_cupt), str(prediction_cupt), str(out_file)],
        cwd=PHD_CODE,
    )


def ensure_dev_config(lang: str) -> Path:
    """
    Create configs/dev_config_{lang}.json from test_config_{lang}.json
    (mode TEST → DEV) unless it already exists.  Returns the config path.
    """
    dev_cfg_path  = MTLB_CODE / "configs" / f"dev_config_{lang}.json"
    test_cfg_path = MTLB_CODE / "configs" / f"test_config_{lang}.json"

    if dev_cfg_path.exists():
        print(f"[INFO] {dev_cfg_path.name} already exists, skipping creation.")
        return dev_cfg_path

    if not test_cfg_path.exists():
        sys.exit(f"test_config not found: {test_cfg_path}")

    cfg = load_config(test_cfg_path)
    cfg["mode"] = "DEV"
    save_config(dev_cfg_path, cfg)
    print(f"[INFO] Created {dev_cfg_path}")
    return dev_cfg_path


# ---------------------------------------------------------------------------
# Per-language pipeline (one run)
# ---------------------------------------------------------------------------

def run_once(lang: str, counter: int) -> None:
    print(f"\n{'='*60}")
    print(f"  LANG={lang}  counter={counter}")
    print(f"{'='*60}")

    test_cfg_path = MTLB_CODE / "configs" / f"test_config_{lang}.json"
    if not test_cfg_path.exists():
        sys.exit(f"Config not found: {test_cfg_path}")

    # ------------------------------------------------------------------
    # 1. Train on TRAINDEV  (main.py with TEST config)
    # ------------------------------------------------------------------
    run(["uv", "run", "main.py", str(test_cfg_path.relative_to(MTLB_CODE))],
        cwd=MTLB_CODE)

    # ------------------------------------------------------------------
    # 2. Predict test set from TRAINDEV model
    # ------------------------------------------------------------------
    test_cfg = load_config(test_cfg_path)
    x_dir = saved_folder_name(test_cfg, "TEST")
    if not x_dir.exists():
        sys.exit(f"Expected saved dir not found: {x_dir}")

    run(["uv", "run", "load_test.py", str(x_dir) + "/", "TEST"],
        cwd=MTLB_CODE)

    pred_traindev = x_dir / "test.mtlb_trained_on_TRAINDEV.cupt"
    if not pred_traindev.exists():
        sys.exit(f"Prediction file not found: {pred_traindev}")

    # ------------------------------------------------------------------
    # 3. Generate dcupt for TRAINDEV prediction
    # ------------------------------------------------------------------
    generate_dcupt(lang, pred_traindev, "TRAINDEV", counter)

    # ------------------------------------------------------------------
    # 4. Ensure dev config exists (TEST→DEV, kept for future runs)
    # ------------------------------------------------------------------
    dev_cfg_path = ensure_dev_config(lang)

    # ------------------------------------------------------------------
    # 5. Delete TRAINDEV model folder
    # ------------------------------------------------------------------
    print(f"[INFO] Deleting {x_dir}")
    shutil.rmtree(x_dir)

    # ------------------------------------------------------------------
    # 6. Train on TRAIN  (main.py with DEV config)
    # ------------------------------------------------------------------
    run(["uv", "run", "main.py", str(dev_cfg_path.relative_to(MTLB_CODE))],
        cwd=MTLB_CODE)

    # ------------------------------------------------------------------
    # 7. Predict test set from TRAIN model
    # ------------------------------------------------------------------
    dev_cfg = load_config(dev_cfg_path)
    y_dir = saved_folder_name(dev_cfg, "DEV")
    if not y_dir.exists():
        sys.exit(f"Expected saved dir not found: {y_dir}")

    run(["uv", "run", "load_test.py", str(y_dir) + "/", "TEST"],
        cwd=MTLB_CODE)

    pred_train = y_dir / "test.mtlb_trained_on_TRAIN.cupt"
    if not pred_train.exists():
        sys.exit(f"Prediction file not found: {pred_train}")

    # ------------------------------------------------------------------
    # 8. Generate dcupt for TRAIN prediction
    # ------------------------------------------------------------------
    generate_dcupt(lang, pred_train, "TRAIN", counter)

    # ------------------------------------------------------------------
    # 9. Delete TRAIN model folder
    # ------------------------------------------------------------------
    print(f"[INFO] Deleting {y_dir}")
    shutil.rmtree(y_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MTLB pipeline N times per language.")
    parser.add_argument("languages", nargs="+",
                        help="Language codes, e.g. DE FR HE")
    args = parser.parse_args()

    for lang in args.languages:
        # Determine starting counter from existing files (across both suffixes,
        # take the max so both TRAINDEV and TRAIN files share the same counter
        # within a run).
        start = max(
            next_counter(lang, "TRAINDEV"),
            next_counter(lang, "TRAIN"),
        )
        for i in range(NUM_RUNS):
            run_once(lang, start + i)


if __name__ == "__main__":
    main()
