
1 - run generate_all_blind.py on parseme/<version>/
2 - run generate_all_gold_dcupt.py on parseme/<version>/
3 - run generate_multi_dcupt.py on parseme/<version>/
	if parseme/<version> is 1.2, exclude RO and EU
uv venv --python 3.9 && uv sync --extra gpu

uv venv --python 3.13 && uv sync --extra gpu

uv venv --python 3.13 && uv sync --extra cpu
