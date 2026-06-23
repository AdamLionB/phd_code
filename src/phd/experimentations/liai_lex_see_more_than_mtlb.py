# from a import data2score_pipeline
# %%
from phd import lexicon, cupt_parser, lcss_lex, utils, oa_indices, lambdacss, scripts, liai, experimentations
import torch
import pandas as pd
import os
from time import perf_counter
from collections import Counter
from pathlib import Path
from scipy.stats import norm, f_oneway, normaltest, ttest_rel, pearsonr, wilcoxon, friedmanchisquare, normaltest, monte_carlo_test, ttest_ind
import itertools as it
import numpy as np
from typing import Optional, Literal

# ── Configuration ──────────────────────────────────────────────────────────────
LANGS = ['DE', 'EL', 'FR',  'GA', 'HE', 'HI',
		'IT', 'PL', 'PT', 'SV', 'TR', 'ZH'
		]
METRICS = ['p', 'r', 'f', 'pred', 'truth', 'tp']
N = 10  # number of MTLB runs per language (indices 1..N-1)
METHOD = 'LIAI'

PARSEME_PATH = os.path.join('..', 'data', 'parseme')
PARSEME_VERSION = '1.2'
TRAINDEV_CORPUS_NAME = 'traindev'
TEST_CORPUS_NAME = 'test'
LEX_PATH = os.path.join('..', 'data', 'lexicons')
MTLB_PRED_PATH = os.path.join('..', 'data', 'mtlb_pred')
PRED_PATH = os.path.join('..', 'data', 'lex_pred')

#%%
# ── Load LIAI model once (language-agnostic multilingual BERT) ─────────────────
if METHOD == 'LIAI':
	LIAI_MODEL_PATH = os.path.join('..', 'data', 'liai')
	LIAI_MODEL_NAME = '45_merger_masked.model'
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	FROZEN = False
	# BERT_CUSTOM_CACHE = str(Path('~/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/fdfce55e83dbed325647a63e7e1f5de19f0382ba').expanduser().resolve())
	BERT_CUSTOM_CACHE = None

	print('Loading LIAI model...')
	liai_model = liai.Merger_with_padding_mask(4, FROZEN, DEVICE, BERT_CUSTOM_CACHE).to(DEVICE)
	state_dict = torch.load(
		os.path.join(LIAI_MODEL_PATH, LIAI_MODEL_NAME),
		map_location=torch.device('cpu') if DEVICE == 'cpu' else None
	)
	try:
		liai_model.load_state_dict(state_dict)
	except Exception:
		state_dict.pop('bert.embedding.embeddings.position_ids', None)
		liai_model.load_state_dict(state_dict)
	print('LIAI model loaded.')

# ── Main loop: one language at a time ──────────────────────────────────────────
records = []  # accumulated score rows → DataFrame at the end

for LANG in LANGS:
	print(f'\n{"="*60}\nProcessing language: {LANG}\n{"="*60}')

	# -- traindev corpus + lexicon -------------------------------------------------
	TRAINDEV_CORPUS_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TRAINDEV_CORPUS_NAME}.gold.dcupt')
	print(f'Loading traindev corpus from {TRAINDEV_CORPUS_PATH}')
	t_start = perf_counter()
	TT_traindev, df_traindev = cupt_parser.setup_data(TRAINDEV_CORPUS_PATH)
	print(f'Corpus loaded in {(perf_counter() - t_start):.2f}s: {len(TT_traindev)} sentences, {len(df_traindev)} tokens')

	LEX_JSON_PATH = os.path.join(LEX_PATH, f'tmp_{LANG}_{"".join(PARSEME_VERSION.split("."))}_{TRAINDEV_CORPUS_NAME}.json')
	print('Generating or loading lexicon...')
	lex_train = scripts.extract_lem_dep_css_lex.generate_or_load_lexicon_from_corpus_TT(TT_traindev, LEX_JSON_PATH)
	mwe_type_dist_train = Counter(lcss.get_mwe_type() for lcss in lex_train)
	print(f'Lexicon: {sum(mwe_type_dist_train.values())} entries, {len(mwe_type_dist_train)} unique MWE types')

	# -- train / dev corpora (to define "seen by" sets) ----------------------------
	df_train = cupt_parser.setup_data_noTT(
		os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, 'train.gold.dcupt')
	)
	train_inline_mwes = cupt_parser.inline_mwes(cupt_parser.get_mwes(df_train))

	df_dev = cupt_parser.setup_data_noTT(
		os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, 'dev.gold.dcupt')
	)
	dev_inline_mwes = cupt_parser.inline_mwes(cupt_parser.get_mwes(df_dev))

	# -- test corpus (gold) --------------------------------------------------------
	TEST_CORPUS_GOLD_PATH  = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.gold.dcupt')
	TEST_CORPUS_BLIND_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.blind.cupt')

	print(f'Loading test corpus from {TEST_CORPUS_GOLD_PATH}')
	t_start = perf_counter()
	TT_test, df_test = cupt_parser.setup_data(TEST_CORPUS_GOLD_PATH)
	test_mwes = cupt_parser.get_mwes(df_test)
	test_inline_mwes = cupt_parser.inline_mwes(test_mwes)
	print(f'Test corpus loaded in {(perf_counter() - t_start):.2f}s: {len(TT_test)} sentences, {len(test_inline_mwes)} MWEs')

	# -- lexicon predictions on test -----------------------------------------------
	LEX_PRED_NAME  = f'{TEST_CORPUS_NAME}.lex_lem_dep_css_trained_on_{TRAINDEV_CORPUS_NAME}.dcupt'
	PRED_FILE_PATH = os.path.join(PRED_PATH, PARSEME_VERSION, LANG, LEX_PRED_NAME)

	if os.path.exists(PRED_FILE_PATH):
		print(f'Loading lex predictions from {PRED_FILE_PATH}')
		df_lex_pred = cupt_parser.setup_data_noTT(PRED_FILE_PATH)
		lex_pred_mwes = cupt_parser.get_mwes(df_lex_pred)
	else:
		print('Matching lexicon against test corpus...')
		t_start = perf_counter()
		lex_pred_mwes = lex_train.match(df_test)
		cupt_parser.write_matches_as_dcupt(lex_pred_mwes, TEST_CORPUS_BLIND_PATH, PRED_FILE_PATH)
		df_lex_pred = cupt_parser.setup_data_noTT(PRED_FILE_PATH)
		print(f'Done in {(perf_counter() - t_start):.2f}s: {len(cupt_parser.inline_mwes(lex_pred_mwes))} predicted MWEs')

	# -- "seen by" sets ------------------------------------------------------------
	seen_by_lex    = set(dev_inline_mwes[0])
	seen_by_system = set(train_inline_mwes[0])
	seen_by_lex_not_by_system = list(seen_by_lex - seen_by_system)   # lex-only
	seen_by_system_not_by_lex = list(seen_by_system - seen_by_lex)   # system-only
	seen_by_lex_and_system    = list(seen_by_lex & seen_by_system)   # both
	seen_by_lex_or_system     = list(seen_by_lex | seen_by_system)   # either (complement = neither)

	# -- loop over MTLB runs -------------------------------------------------------
	for n in range(1, N):
		SYSTEM_PATH = os.path.join(MTLB_PRED_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.mtlb_trained_on_TRAIN_{n}.dcupt')
		print(f'[{LANG} run {n}] Loading system predictions from {SYSTEM_PATH}')
		t_start = perf_counter()
		_, df_system_pred = cupt_parser.setup_data(SYSTEM_PATH)
		system_pred_mwes = cupt_parser.get_mwes(df_system_pred)
		print(f'[{LANG} run {n}] System loaded in {(perf_counter() - t_start):.2f}s: {len(cupt_parser.inline_mwes(system_pred_mwes))} predicted MWEs')

		# -- merge -----------------------------------------------------------------
		print(f'[{LANG} run {n}] Merging with {METHOD}...')
		t_start = perf_counter()
		if METHOD == 'union':
			merged_pred_mwes = cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes)
		elif METHOD == 'intersection':
			merged_pred_mwes = cupt_parser.intersection_mwes(lex_pred_mwes, system_pred_mwes)
		elif METHOD == 'LIAI':
			LIAI_PRED_PATH = os.path.join('..', 'data', 'liai_pred', PARSEME_VERSION, LANG)
			os.makedirs(LIAI_PRED_PATH, exist_ok=True)
			_model_stem = LIAI_MODEL_NAME.removesuffix('.model')
			_blind_stem = os.path.basename(TEST_CORPUS_BLIND_PATH).removesuffix('.cupt').removesuffix('.blind')
			LIAI_PRED_FILE = (
				f'{_blind_stem}'
				f'.liai_{_model_stem}'
				f'.sys_TRAIN_{n}'
				f'.lex_{TRAINDEV_CORPUS_NAME}'
				f'.dcupt'
			)
			LIAI_PRED_FILE_PATH = os.path.join(LIAI_PRED_PATH, LIAI_PRED_FILE)

			if os.path.exists(LIAI_PRED_FILE_PATH):
				print(f'[{LANG} run {n}] Loading cached LIAI predictions from {LIAI_PRED_FILE_PATH}')
				df_merged_pred = cupt_parser.setup_data_noTT(LIAI_PRED_FILE_PATH)
				merged_pred_mwes = cupt_parser.get_mwes(df_merged_pred)
			else:
				voc = liai.prep.Voc()
				test_sentences, _, _, _, _, Y_system_test, _, _, _, _, _ = liai.prep.df2ts(df_system_pred, voc, 0, 1)
				_, _, _, _, _, Y_lex_test, _, _, _, _, _ = liai.prep.df2ts(df_lex_pred, voc, 0, 1)
				ts_pred = liai.predict(liai_model, test_sentences, Y_system_test, Y_lex_test, device=DEVICE, batch_size=30)
				merged_pred_mwes = liai.prep.ts2df(
					df_test,
					cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes),
					ts_pred
				)
				print(f'[{LANG} run {n}] Saving LIAI predictions to {LIAI_PRED_FILE_PATH}')
				cupt_parser.write_matches_as_dcupt(merged_pred_mwes, TEST_CORPUS_BLIND_PATH, LIAI_PRED_FILE_PATH)
		print(f'[{LANG} run {n}] Merged in {(perf_counter() - t_start):.2f}s: {len(cupt_parser.inline_mwes(merged_pred_mwes))} predicted MWEs')

		# -- score -----------------------------------------------------------------
		scores_merged_lex_only, _  = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_lex_not_by_system)
		scores_merged_both,     _  = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_lex_and_system)
		scores_merged_system_only, _ = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_system_not_by_lex)
		_, scores_merged_neither   = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_lex_or_system)

		scores_merged_all          = oa_indices.full_eval(test_mwes, merged_pred_mwes)

		scores_system_lex_only, _  = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_lex_not_by_system)
		scores_system_both,     _  = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_lex_and_system)
		scores_system_system_only, _ = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_system_not_by_lex)
		_, scores_system_neither   = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_lex_or_system)
		scores_system_all          = oa_indices.full_eval(test_mwes, system_pred_mwes)

		group_scores = {
			('merged', 'lex_only'): scores_merged_lex_only,
			('merged', 'both'):     scores_merged_both,
			('merged', 'system_only'): scores_merged_system_only,
			('merged', 'neither'):  scores_merged_neither,
			('merged', 'all'):      scores_merged_all,
			('system', 'lex_only'): scores_system_lex_only,
			('system', 'both'):     scores_system_both,
			('system', 'system_only'): scores_system_system_only,
			('system', 'neither'):  scores_system_neither,
			('system', 'all'):      scores_system_all,
		}
		for (method, group), scores in group_scores.items():
			for metric in METRICS:
				records.append({
					'lang':   LANG,
					'run':    n,
					'method': method,
					'group':  group,
					'metric': metric,
					'value':  scores.get(metric),
				})

# ── Build DataFrame and aggregate ─────────────────────────────────────────────
# %%
df_scores = pd.DataFrame(records)
df_scores['value'] = pd.to_numeric(df_scores['value'], errors='coerce')

# Per-language mean/std across runs
df_lang = (
	df_scores
	.groupby(['lang', 'method', 'group', 'metric'])['value']
	.agg(mean='mean', std='std')
	.reset_index()
)

# ── Micro-average multilingual scores ─────────────────────────────────────────
# For each run n, sum tp/pred/truth across all languages, then recompute p/r/f.
# This gives one score per run, preserving pairing for t-tests.
_count_cols = ['tp', 'pred', 'truth']
_df_counts = (
	df_scores[df_scores['metric'].isin(_count_cols)]
	.groupby(['run', 'method', 'group', 'metric'])['value']
	.sum()
	.reset_index()
)
_df_wide = _df_counts.pivot_table(
	index=['run', 'method', 'group'], columns='metric', values='value'
).reset_index()
_df_wide['p'] = _df_wide['tp'] / _df_wide['pred']
_df_wide['r'] = _df_wide['tp'] / _df_wide['truth']
_df_wide['f'] = 2 * _df_wide['p'] * _df_wide['r'] / (_df_wide['p'] + _df_wide['r'])

df_multi = _df_wide.melt(
	id_vars=['run', 'method', 'group'],
	value_vars=['p', 'r', 'f', 'tp', 'pred', 'truth'],
	var_name='metric', value_name='value'
)
df_multi['lang'] = '[multilingual]'

# Unified frame — single source of truth for all helpers
df_scores_all = pd.concat([df_scores, df_multi], ignore_index=True)

# Per-(method, group, metric) mean/std for multilingual
df_multi_lang = (
	df_multi
	.groupby(['method', 'group', 'metric'])['value']
	.agg(mean='mean', std='std')
	.reset_index()
)

#%%
# n_bootstrap = 10000
# df_bootstrap = pd.DataFrame(columns=['lang', 'method', 'group', 'metric', 'fscore'])
# aggregate_fscores = []
# for i in range(1, 10):
# 	for (group, method), x in df_scores_all.loc[lambda x: x['lang'] != '[multilingual]'].groupby(['group', 'method']):
# 		agg = []
# 		for lang, y in x.groupby('lang'):
# 			r = y.sample(1000, replace=True)['run'].reset_index(drop=True)
# 			z = y.set_index('run').groupby('metric').sum(numeric_only=True)
# 			precision = z.loc['tp'].value / z.loc['pred'].value 
# 			recall = z.loc['tp'].value / z.loc['truth'].value 

# 			f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
# 			aggregate_fscores.append(['bootstrap', i, method, group, 'f', f])
# 		# break
# df_bootstrap = pd.DataFrame(aggregate_fscores, columns=['lang', 'run', 'method', 'group', 'metric', 'value'])

def bootstrap_multilingual(df, n_pseudo_runs=10, runs_per_pseudo=30, seed=42):
	rng = np.random.default_rng(seed)

	raw = df[df["metric"].isin(["pred", "truth", "tp"])]

	pivoted = (
		raw
		.pivot_table(
			index=["lang", "run", "method", "group"],
			columns="metric",
			values="value",
			aggfunc="first",
		)
		.reset_index()
	)

	results = []

	for i in range(1, n_pseudo_runs):
		sampled_rows = []

		# Sample run IDs per language only — method/group come along for free
		for lang, lang_group in pivoted.groupby("lang"):
			available_runs = lang_group["run"].unique()
			chosen_runs = rng.choice(available_runs, size=runs_per_pseudo, replace=True)

			for run_id in chosen_runs:
				sampled_rows.append(lang_group[lang_group["run"] == run_id])

		sample = pd.concat(sampled_rows, ignore_index=True)

		# Now aggregate by method and group, computing micro-averaged P/R for each cell
		agg = (
			sample
			.groupby(["method", "group"])[["pred", "truth", "tp"]]
			.sum()
			.reset_index()
		)
		agg["lang"] = '[bootstrap]'
		agg["run"] = i
		agg["p"] = agg["tp"] / agg["pred"].replace(0, float("nan"))
		agg["r"] = agg["tp"] / agg["truth"].replace(0, float("nan"))
		agg["f"] = 2 * agg["p"] * agg["r"] / (agg["p"] + agg["r"]).replace(0, float("nan"))

		results.append(agg)

	results_df = pd.concat(results, ignore_index=True)

	# Summary: mean and std across the 10 pseudo-runs, per (method, group)
	# summary = (
	# 	results_df
	# 	.groupby(["method", "group"])[["p", "r", "f"]]
	# 	.agg(["mean", "std"])
	# )
	# summary.columns = ["_".join(c) for c in summary.columns]
	# summary = summary.reset_index()
	return results_df.melt(id_vars=["lang", "run", "method", "group"], value_vars=["p", "r", "f", "pred", "truth", "tp"], var_name="metric")
	# return results_df, summary


bootstrap_df= bootstrap_multilingual(df_scores_all.loc[lambda x: x['lang'] != '[multilingual]'])
df_scores_all = pd.concat([df_scores_all, bootstrap_df], ignore_index=True)

# df_scores_all = pd.concat([df_scores_all, df_bootstrap], ignore_index=True)

# for _ in range(n_bootstrap):
#     # For each language, pick one repetition at random
#     tp_total = sum(np.random.choice(tp_lang) for tp_lang in all_languages_tp)
#     fp_total = sum(np.random.choice(fp_lang) for fp_lang in all_languages_fp)
#     fn_total = sum(np.random.choice(fn_lang) for fn_lang in all_languages_fn)
    
#     f = 2*tp_total / (2*tp_total + fp_total + fn_total)
#     aggregate_fscores.append(f)

#%%
fmt = lambda x: f'{x:.4f}' if pd.notna(x) else 'nan'

# ── Shared helpers ─────────────────────────────────────────────────────────────
# from scipy.stats import norm as _norm

def gaussian_sup(mean_1, std_1, mean_2, std_2):
	"""P(X > Y) where X\~N(mean_1,std_1²), Y\~N(mean_2,std_2²)."""
	denom = (std_1**2 + std_2**2) ** 0.5
	if denom == 0:
		return float('nan')
	return norm(loc=mean_1 - mean_2, scale=denom).sf(0)

def paired_ttest_df(df, lang, method1, group1, method2, group2, metric):
	"""Returns (mean_delta, p_value) for paired t-test on the run series."""
	a = df[(df['lang'] == lang) & (df['method'] == method1) & (df['group'] == group1) & (df['metric'] == metric)].set_index('run')['value']
	b = df[(df['lang'] == lang) & (df['method'] == method2) & (df['group'] == group2) & (df['metric'] == metric)].set_index('run')['value']
	shared = a.index.intersection(b.index)
	pairs = [(a[i], b[i]) for i in shared if pd.notna(a[i]) and pd.notna(b[i])]
	if len(pairs) < 2:
		return float('nan'), float('nan')
	v1, v2 = zip(*pairs)
	_, p = ttest_rel(v1, v2)
	return float(sum(x - y for x, y in pairs) / len(pairs)), p

def get_mean_std(lang, method, group, metric):
	"""Mean and std for a (lang, method, group, metric) cell."""
	if lang == '[multilingual]':
		row = df_multi_lang[(df_multi_lang['method'] == method) & (df_multi_lang['group'] == group) & (df_multi_lang['metric'] == metric)]
	else:
		row = df_lang[(df_lang['lang'] == lang) & (df_lang['method'] == method) & (df_lang['group'] == group) & (df_lang['metric'] == metric)]
	if len(row) == 0:
		return float('nan'), float('nan')
	return float(row['mean'].iloc[0]), float(row['std'].iloc[0])

# ── Key comparison tables ──────────────────────────────────────────────────────
# print('\n\n' + '='*70)
# print('KEY COMPARISONS')
# print('merged[lex_only] vs merged[neither]  |  system[lex_only] vs system[neither] vs system[both]')
# print('Sanity: system[lex_only] ≈ system[neither] (both unseen by system training)')
# print('='*70)

ALL_LANGS = LANGS + ['[multilingual]', '[bootstrap]']

key_combos = [
	('merged', 'lex_only'),
	('merged', 'system_only'),
	('merged', 'neither'),
	('system', 'lex_only'),
	('system', 'system_only'),
	('system', 'neither'),
	('system', 'both'),
]

# for metric in METRICS:
# 	print(f'\n── Metric: {metric} ──')
# 	rows = []
# 	for lang in ALL_LANGS:
# 		row = {'lang': lang}
# 		for (method, group) in key_combos:
# 			mean, _ = get_mean_std(lang, method, group, metric)
# 			row[f'{method}_{group}'] = mean
# 		rows.append(row)

# 	df_table = pd.DataFrame(rows).set_index('lang')
# 	df_table['Δmerged(lex_only−neither)'] = df_table['merged_lex_only'] - df_table['merged_neither']
# 	df_table['Δsystem(lex_only−neither)'] = df_table['system_lex_only'] - df_table['system_neither']
# 	print(df_table.to_string(float_format=lambda x: f'{x:.4f}'))

# ── Paired t-test: merged[lex_only] vs merged[neither] ────────────────────────
# print('\n\n' + '='*70)
# print('PAIRED T-TEST (across runs): merged[lex_only] vs merged[neither]')
# print('='*70)

# for metric in ['f', 'p', 'r']:
# 	print(f'\n── Metric: {metric} ──')
# 	ttest_rows = []
# 	for lang in ALL_LANGS:
# 		delta_merged, p_merged = paired_ttest_df(df_scores_all, lang, 'merged', 'lex_only', 'merged', 'neither', metric)
# 		delta_system, p_system = paired_ttest_df(df_scores_all, lang, 'system', 'lex_only', 'system', 'neither', metric)
# 		ttest_rows.append({
# 			'lang':           lang,
# 			'Δmerged mean':   delta_merged,
# 			'p merged':       p_merged,
# 			'Δsystem mean':   delta_system,
# 			'p system':       p_system,
# 		})
# 	df_ttest = pd.DataFrame(ttest_rows).set_index('lang')
# 	print(df_ttest.to_string(float_format=lambda x: f'{x:.4f}'))

# ── Non-parametric helpers ────────────────────────────────────────────────────
# def _wilcoxon_aligned(lang, method1, group1, method2, group2, metric, alternative='two-sided'):
# 	"""Returns (mean_delta, p_value) via Wilcoxon signed-rank test."""
# 	a = df_scores_all[(df_scores_all['lang'] == lang) & (df_scores_all['method'] == method1) &
# 	                  (df_scores_all['group'] == group1) & (df_scores_all['metric'] == metric)].set_index('run')['value']
# 	b = df_scores_all[(df_scores_all['lang'] == lang) & (df_scores_all['method'] == method2) &
# 	                  (df_scores_all['group'] == group2) & (df_scores_all['metric'] == metric)].set_index('run')['value']


	
# 	shared = a.index.intersection(b.index)
# 	pairs = [(a[i], b[i]) for i in shared if pd.notna(a[i]) and pd.notna(b[i])]
# 	if len(pairs) < 2:
# 		return float('nan'), float('nan')
# 	v1, v2 = zip(*pairs)
# 	mean_delta = float(sum(x - y for x, y in pairs) / len(pairs))
# 	if all(x == y for x, y in pairs):
# 		return mean_delta, 1.0
# 	try:
# 		v = [round(x - y,4)  for x, y in zip(v1, v2)]
# 		_, p = wilcoxon(v, alternative=alternative)#, zero_method='pratt')
# 	except ValueError:
# 		return mean_delta, float('nan')
# 	return mean_delta, float(p)

# def _friedman_posthoc_mixed(lang, method_group_pairs, metric):
# 	"""Friedman test across arbitrary (method, group) pairs, then pairwise Wilcoxon
# 	with Bonferroni correction over all pairs tested.
# 	Returns (friedman_p, {((m1,g1),(m2,g2)): (mean_delta, bonferroni_p)})."""
# 	series = {}
# 	for mg in method_group_pairs:
# 		m, g = mg
# 		s = df_scores_all[
# 			(df_scores_all['lang'] == lang) & (df_scores_all['method'] == m) &
# 			(df_scores_all['group'] == g) & (df_scores_all['metric'] == metric)
# 		].set_index('run')['value']
# 		series[mg] = s.dropna()
# 	common_idx = series[method_group_pairs[0]].index
# 	for mg in method_group_pairs[1:]:
# 		common_idx = common_idx.intersection(series[mg].index)
# 	if len(common_idx) < 3:
# 		return float('nan'), {}
# 	aligned = {mg: series[mg][common_idx].values for mg in method_group_pairs}
# 	try:
# 		_, fp = friedmanchisquare(*[aligned[mg] for mg in method_group_pairs])
# 	except ValueError:
# 		return float('nan'), {}
# 	n_pairs = len(method_group_pairs) * (len(method_group_pairs) - 1) // 2
# 	pairwise = {}
# 	for i, mg1 in enumerate(method_group_pairs):
# 		for mg2 in method_group_pairs[i + 1:]:
# 			v1, v2 = aligned[mg1], aligned[mg2]
# 			mean_d = float((v1 - v2).mean())
# 			if (v1 == v2).all():
# 				pairwise[(mg1, mg2)] = (mean_d, min(1.0, 1.0 * n_pairs))
# 			else:
# 				try:
# 					_, wp = wilcoxon(v1, v2)
# 					pairwise[(mg1, mg2)] = (mean_d, min(1.0, float(wp) * n_pairs))  # Bonferroni
# 				except ValueError:
# 					pairwise[(mg1, mg2)] = (mean_d, float('nan'))
# 	return float(fp), pairwise, n_pairs

def get_sample(lang, method, group, metric) -> pd.Series:
	filters = df_scores_all['lang'].where(lambda s: (s == s) & (s != s), True) # arrays of Trues
	if lang is not None:
		filters &= df_scores_all['lang'] == lang
	if method is not None:
		filters &= df_scores_all['method'] == method
	if group is not None:
		filters &= df_scores_all['group'] == group
	if metric is not None:
		filters &= df_scores_all['metric'] == metric
	return df_scores_all[filters].set_index('run')['value']

# ── LIAI improvement analysis ─────────────────────────────────────────────────
print('\n\n' + '='*70)
print('LIAI IMPROVEMENT ANALYSIS')
print('Does LIAI+lexicon improve over system alone?')
print()
print('Criterion A | better overall perf | merged_all > system_all ')
print('  T-test two-sided p<0.05 AND T-test greater p<0.05')
print()
print('Criterion B | better results on merged_lex_only')
print('Sanity: system_lex_only ≈ system_neither ')
print('  T-test two-sided p>=0.05 AND T-test greater p<0.05 AND T-test less p<0.05')
print('  → if violated: lex_only easier than neither for system, criterion B1 and B2 not applicable')
print('  → if holds:    ')
print('Criterion B1 | better results on fully unseen by LIAI | merged_lex_only > merged_neither')
print('  T-test two-sided p<0.05 AND T-test greater p<0.05')
print('Criterion B2 | better results on fully unseen by system | merged_lex_only > system_neither')
print('  T-test two-sided p<0.05 AND T-test greater p<0.05')
print('Criterion B3 | better results than system on seen only by lex | merged_lex_only > system_lex_only')
print('  T-test two-sided p<0.05 AND T-test greater p<0.05')
print('Criterion C | better results on seen by both | merged_both > system_both')
print('  T-test two-sided p<0.05 AND T-test greater p<0.05')
print('Criterion D | similar results on seen only by system | merged_system_only > system_system_only')
print('  T-test two-sided p>=0.05 AND T-test greater p<0.05 AND T-test less p<0.05')

# print('S2: merged_neither NOT significantly < system_neither  (no harmful tradeoff)')
print('='*70)

ALPHA = 0.05

res = pd.DataFrame(columns=['language', 'a', 'b', 'hypothesis', 'null_rejected', 'a is normal', 'b is normal'])

summary = []
for lang in ALL_LANGS:
	# print(f'\n── {lang} ──')
	metric = 'f'

	s_ma = get_sample(lang, 'merged', 'all', 'f').rename('merged_all')
	s_ml = get_sample(lang, 'merged', 'lex_only', 'f').rename('merged_lex_only')
	s_mn = get_sample(lang, 'merged', 'neither', 'f').rename('merged_neither')
	s_mb = get_sample(lang, 'merged', 'both', 'f').rename('merged_both')
	s_ms = get_sample(lang, 'merged', 'system_only', 'f').rename('merged_system_only')
	s_sa = get_sample(lang, 'system', 'all', 'f').rename('system_all')
	s_sl = get_sample(lang, 'system', 'lex_only', 'f').rename('system_lex_only')
	s_sn = get_sample(lang, 'system', 'neither', 'f').rename('system_neither')
	s_sb = get_sample(lang, 'system', 'both', 'f').rename('system_both')
	s_ss = get_sample(lang, 'system', 'system_only', 'f').rename('system_system_only')


	# print('----------')
	# from scipy.stats import normaltest
	# print(normaltest(a))
	# print('----------')

	# mean_ma, std_ma = s_ma.mean(), s_ma.std()
	# mean_sa, std_sa = s_sa.mean(), s_sa.std()
	# mean_ml, std_ml = s_ml.mean(), s_ml.std()
	# mean_mn, std_mn = s_mn.mean(), s_mn.std()
	# mean_mb, std_mb = s_mb.mean(), s_mb.std()
	# mean_ms, std_ms = s_ms.mean(), s_ms.std()
	# mean_sl, std_sl = s_sl.mean(), s_sl.std()
	# mean_sn, std_sn = s_sn.mean(), s_sn.std()
	# mean_sb, std_sb = s_sb.mean(), s_sb.std()
	# mean_ss, std_ss = s_ss.mean(), s_ss.std()
	# Criterion A — Wilcoxon: merged_all vs system_all
	# delta_ma_sa, pv_ma_neq_sa = _wilcoxon_aligned(lang, 'merged', 'all', 'system', 'all', metric)
	# # prob_ma_gt_sa  = gaussian_sup(mean_ma, std_ma, mean_sa, std_sa)
	# crit_ma_gt_sa  = (delta_ma_sa > 0) and pd.notna(pv_ma_neq_sa) and (pv_ma_neq_sa < ALPHA)

	# # Criterion B — 4-way Friedman: merged_lex_only, merged_neither, system_lex_only, system_neither
	# # All 6 pairwise posthoc Wilcoxon tests are Bonferroni-corrected within this family.
	# _crit_b_pairs = [
	# 	('merged', 'lex_only'), ('merged', 'neither'),
	# 	('system', 'lex_only'), ('system', 'neither'),
	# ]
	# friedman_p, posthoc, n_pairs = _friedman_posthoc_mixed(lang, _crit_b_pairs, metric)



	# _key_sl_sn = (('system', 'lex_only'), ('system', 'neither'))  # data in lex only not easier for system (sanity check)
	# _key_ml_sl  = (('merged', 'lex_only'), ('system', 'lex_only')) # S1: genuine uplift
	# _key_mn_sn  = (('merged', 'neither'),  ('system', 'neither'))  # S2: no harmful tradeoff
	
	# _key_b   = (('merged', 'lex_only'), ('merged', 'neither'))  # main claim

	# delta_b,   p_b   = posthoc.get(_key_b,   (float('nan'), float('nan')))
	# # delta_sl_sn,  pv_sl_sn = posthoc.get(_key_sl_sn, (float('nan'), float('nan')))
	# delta_sl_sn,  pv_sl_neq_sn = _wilcoxon_aligned(lang, 'system', 'lex_only', 'system', 'neither', metric)
	# # delta_ml_sl,  pv_ml_sl  = posthoc.get(_key_ml_sl,  (float('nan'), float('nan')))
	# delta_ml_sl,  pv_ml_neq_sl = _wilcoxon_aligned(lang, 'merged', 'lex_only', 'system', 'lex_only', metric)
	# # delta_mn_sn,  pv_mn_sn  = posthoc.get(_key_mn_sn,  (float('nan'), float('nan')))
	# delta_mn_sn,  pv_mn_neq_sn = _wilcoxon_aligned(lang, 'merged', 'neither', 'system', 'neither', metric)


	# f_delta_gt_0 = lambda d: d > 0
	f_null_rejected = lambda p : pd.notna(p) and (p < ALPHA)
	f_null_not_rejected = lambda p : pd.notna(p) and (p >= ALPHA)
	# f_a_gt_b = lambda d_a_b, p_a_b: (d_a_b > 0) and f_null_rejected(p_a_b)
	# f_a_lt_b = lambda d_a_b, p_a_b: (d_a_b < 0) and f_null_rejected(p_a_b)
	# f_a_eq_b = lambda p_a_b: f_null_not_rejected(p_a_b)


	# # prob_b  = gaussian_sup(mean_ml, std_ml, mean_mn, std_mn)
	# # prob_ml_sl = gaussian_sup(mean_ml, std_ml, mean_sl, std_sl)

	# friedman_sig    = pd.notna(friedman_p) and (friedman_p < ALPHA)
	# crit_b          = friedman_sig and (delta_b > 0) and pd.notna(p_b) and (p_b < ALPHA)
	# # Selection bias: system sees lex_only and neither as same difficulty
	# s_sl_eq_sn = f_a_eq_b(pv_sl_neq_sn)
	
	# s_sl_ls_sn = (delta_sl_sn < 0) and pd.notna(pv_sl_neq_sn) and (pv_sl_neq_sn < ALPHA)

	# s_sl_gt_sn = (delta_sl_sn > 0) and pd.notna(pv_sl_neq_sn) and (pv_sl_neq_sn < ALPHA)
	# # 


	# # S1: merged lex_only > above system lex_only
	# s_ml_gt_sl           = (delta_ml_sl > 0) and f_null_rejected(pv_ml_neq_sl)
	# # S2: merged does NOT significantly hurt neither (tradeoff)
	# #   fail = merged_neither significantly BELOW system_neither
	# s2_harmful      = (delta_mn_sn < 0) and pd.notna(pv_mn_neq_sn) and (pv_mn_neq_sn < ALPHA)
	# s2_ok           = not s2_harmful


	# 1 - check if s_sl_eq_sn or s_sl_ls_sn (sanity)
	# 1.1 if yes then check s_ml_gt_sn
	# 1.2 if no  then s_sl_gt_sn should be true (check)
	# 1.2.1 if yes then check s_ml_gt_sl
	# 1.2.1.1 if yes HOLDS
	# 1.2.1.2 if no then NOT MET
	# 1.2.2 if no then s_sl_lt_sn should be true
	
	samples = [s_ma, s_ml, s_mn, s_mb, s_ms, s_sa, s_sl, s_sn, s_sb, s_ss]
	
	normality_results = {}
	for s in samples:
		normality_results[s.name] = experimentations.reject(experimentations.normality_test(s))


	hypothesis_results = []
	
	for a, b in it.combinations(samples, 2):

		p_under_null_a_eq_b = experimentations.paired_equality_test(a, b, 'two-sided', normality_results[a.name], normality_results[b.name])
		p_under_null_a_le_b = experimentations.paired_equality_test(a, b, 'greater', normality_results[a.name], normality_results[b.name])
		p_under_null_a_ge_b = experimentations.paired_equality_test(a, b, 'less', normality_results[a.name], normality_results[b.name])


		hypothesis_results.append([lang, a.name, b.name, 'null_a_eq_b', experimentations.reject(p_under_null_a_eq_b), normality_results[a.name], normality_results[b.name]])
		hypothesis_results.append([lang, a.name, b.name, 'null_a_le_b', experimentations.reject(p_under_null_a_le_b), normality_results[a.name], normality_results[b.name]])
		hypothesis_results.append([lang, a.name, b.name, 'null_a_ge_b', experimentations.reject(p_under_null_a_ge_b), normality_results[a.name], normality_results[b.name]])


	
	tmp_res = pd.DataFrame(hypothesis_results, columns=['language', 'a', 'b', 'hypothesis', 'null_rejected', 'a is normal', 'b is normal'])
	res = pd.concat([res, tmp_res], ignore_index=True)
	
	
	
	
	def validate_hypothesis(
		pos_criterion : Optional[experimentations.Flag],
		neg_criterion : Optional[experimentations.Flag]
	):
		if pos_criterion is not None and pos_criterion == experimentations.REJECTED:
			return experimentations.POSITIVE
		if neg_criterion is not None and neg_criterion == experimentations.REJECTED:
			return experimentations.NEGATIVE
		return experimentations.INCONCLUSIVE




	summary.append([
		lang,
		'criterion A',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_all') & 
				(df['b'] == 'system_all') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_all') & 
				(df['b'] == 'system_all') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0]
		)
	])

	summary.append([
		lang,
		'sanity',
		validate_hypothesis(
			None,
			tmp_res.loc[lambda df: 
				(df['a'] == 'system_lex_only') & 
				(df['b'] == 'system_neither') & 
				(df['hypothesis'] == 'null_a_eq_b')
			]['null_rejected'].values[0],
		)
	])

	summary.append([
		lang,
		'criterion b1',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_lex_only') & 
				(df['b'] == 'merged_neither') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_lex_only') & 
				(df['b'] == 'merged_neither') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0]
		)
		])

	summary.append([
			lang,
			'criterion b2',
			validate_hypothesis(
				tmp_res.loc[lambda df: 
					(df['a'] == 'merged_lex_only') & 
					(df['b'] == 'system_neither') & 
					(df['hypothesis'] == 'null_a_le_b')
				]['null_rejected'].values[0],
				tmp_res.loc[lambda df: 
					(df['a'] == 'merged_lex_only') & 
					(df['b'] == 'system_neither') & 
					(df['hypothesis'] == 'null_a_ge_b')
				]['null_rejected'].values[0]
			)
			])

	
	summary.append([
			lang,
			'criterion b3',
			validate_hypothesis(
				tmp_res.loc[lambda df: 
					(df['a'] == 'merged_lex_only') & 
					(df['b'] == 'system_lex_only') & 
					(df['hypothesis'] == 'null_a_le_b')
				]['null_rejected'].values[0],
				tmp_res.loc[lambda df: 
					(df['a'] == 'merged_lex_only') & 
					(df['b'] == 'system_lex_only') & 
					(df['hypothesis'] == 'null_a_ge_b')
				]['null_rejected'].values[0]
			)
			])

	summary.append([
		lang,
		'criterion C',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_both') & 
				(df['b'] == 'system_both') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_both') & 
				(df['b'] == 'system_both') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0]
		)
	])	

	summary.append([
		lang,
		'criterion D',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_system_only') & 
				(df['b'] == 'system_system_only') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_system_only') & 
				(df['b'] == 'system_system_only') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0],
		)
	])

	
	summary.append([
		lang,
		'criterion E',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_neither') & 
				(df['b'] == 'system_neither') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_neither') & 
				(df['b'] == 'system_neither') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0],
		)
	])

df_summary = pd.DataFrame(summary, columns=['language', 'hypothesis', 'result'])
df_summary


# %%
df_scores_all.to_csv(os.path.join('liai_seen_scores_all.csv'), index=False)

# %%%
df_scores_all = pd.read_csv(os.path.join('liai_seen_scores_all.csv'))

# %% OLD EVAL FOR COMPARISON (kept for reference)

# _,_,_,_,_,Y_truth_test,_,_,_,_,_ = liai.prep.df2ts(df_test, voc, 0, 1)
# truth_test, data_test = liai.build_candidate_table_and_labels(Y_system_test, Y_lex_test, Y_truth_test)
# liai.eval_model(
# 	liai_model,
# 	test_sentences,
# 	Y_truth_test,
# 	data_test,
# 	truth_test,
# 	df_test,
# 	device=DEVICE,
# )
# %%
