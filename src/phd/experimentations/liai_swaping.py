# from a import data2score_pipeline
# %%
from phd import lexicon, cupt_parser, lcss_lex, utils, oa_indices, lambdacss, scripts, liai, experimentations
import torch
import pandas as pd
import os
from time import perf_counter
from collections import Counter
from pathlib import Path
from scipy.stats import norm, f_oneway, normaltest, ttest_rel, pearsonr, wilcoxon, friedmanchisquare, normaltest, monte_carlo_test, ttest_ind, mannwhitneyu

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
	start_time = perf_counter()
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
	end_time = perf_counter()
	print(f'LIAI model loaded in {end_time - start_time:.2f}s.')

# ── Main loop: one language at a time ──────────────────────────────────────────
records = []  # accumulated score rows → DataFrame at the end

for LANG in LANGS:
	print(f'\n{"="*60}\nProcessing language: {LANG}\n{"="*60}')

	# -- traindev corpus + lexicon -------------------------------------------------
	TRAINDEV_CORPUS_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TRAINDEV_CORPUS_NAME}.gold.dcupt')
	print(f'Loading traindev corpus from {TRAINDEV_CORPUS_PATH}')
	t_start = perf_counter()
	TT_traindev, df_traindev = cupt_parser.setup_data(TRAINDEV_CORPUS_PATH)
	end_time = perf_counter()
	print(f'Corpus loaded in {(end_time - t_start):.2f}s: {len(TT_traindev)} sentences, {len(df_traindev)} tokens')

	LEX_JSON_PATH = os.path.join(LEX_PATH, f'tmp_{LANG}_{"".join(PARSEME_VERSION.split("."))}_{TRAINDEV_CORPUS_NAME}.json')
	print('Generating or loading lexicon...')
	t_start = perf_counter()
	lex_train = scripts.extract_lem_dep_css_lex.generate_or_load_lexicon_from_corpus_TT(TT_traindev, LEX_JSON_PATH)
	mwe_type_dist_train = Counter(lcss.get_mwe_type() for lcss in lex_train)
	end_time = perf_counter()
	print(f'Lexicon: {sum(mwe_type_dist_train.values())} entries, {len(mwe_type_dist_train)} unique MWE types, loaded/generated in {(end_time - t_start):.2f}s')

	# -- train / dev corpora (to define "seen by" sets) ----------------------------
	# df_train = cupt_parser.setup_data_noTT(
	# 	os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, 'train.gold.dcupt')
	# )
	# train_inline_mwes = cupt_parser.inline_mwes(cupt_parser.get_mwes(df_train))

	# df_dev = cupt_parser.setup_data_noTT(
	# 	os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, 'dev.gold.dcupt')
	# )
	# dev_inline_mwes = cupt_parser.inline_mwes(cupt_parser.get_mwes(df_dev))

	# -- test corpus (gold) --------------------------------------------------------
	TEST_CORPUS_GOLD_PATH  = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.gold.dcupt')
	TEST_CORPUS_BLIND_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.blind.cupt')

	print(f'Loading test corpus from {TEST_CORPUS_GOLD_PATH}')
	t_start = perf_counter()
	TT_test, df_test = cupt_parser.setup_data(TEST_CORPUS_GOLD_PATH)
	test_mwes = cupt_parser.get_mwes(df_test)
	test_inline_mwes = cupt_parser.inline_mwes(test_mwes)
	end_time = perf_counter()
	print(f'Test corpus loaded in {(end_time - t_start):.2f}s: {len(TT_test)} sentences, {len(test_inline_mwes)} MWEs')

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
		end_time = perf_counter()
		print(f'Done in {(end_time - t_start):.2f}s: {len(cupt_parser.inline_mwes(lex_pred_mwes))} predicted MWEs')

	# -- "seen by" sets ------------------------------------------------------------
	# seen_by_lex    = set(dev_inline_mwes[0])
	# seen_by_system = set(train_inline_mwes[0])
	# seen_by_lex_not_by_system = list(seen_by_lex - seen_by_system)   # lex-only
	# seen_by_system_not_by_lex = list(seen_by_system - seen_by_lex)   # system-only
	# seen_by_lex_and_system    = list(seen_by_lex & seen_by_system)   # both
	# seen_by_lex_or_system     = list(seen_by_lex | seen_by_system)   # either (complement = neither)

	# -- loop over MTLB runs -------------------------------------------------------

	# if method == 'LIAI':
	# 	voc = liai.prep.Voc()
	# 	test_sentences, _, _, _, _, Y_system_test, _, _, _, _, _ = liai.prep.df2ts(df_system_pred, voc, 0, 1)
	# 	_, _, _, _, _, Y_lex_test, _, _, _, _, _ = liai.prep.df2ts(df_lex_pred, voc, 0, 1)

	swaps = [
		('sent', 'sent'),
		('sys', 'lex'),
		('lex', 'sys'),
		('sys', 'sys'),
		('lex', 'lex'),
		('rand', 'rand'),
	]


	
	for n in range(1, N):
		SYSTEM_PATH = os.path.join(MTLB_PRED_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.mtlb_trained_on_TRAINDEV_{n}.dcupt')
		print(f'[{LANG} run {n}] Loading system predictions from {SYSTEM_PATH}')
		t_start = perf_counter()
		_, df_system_pred = cupt_parser.setup_data(SYSTEM_PATH)
		system_pred_mwes = cupt_parser.get_mwes(df_system_pred)
		end_time = perf_counter()
		print(f'[{LANG} run {n}] System loaded in {(end_time - t_start):.2f}s: {len(cupt_parser.inline_mwes(system_pred_mwes))} predicted MWEs')

		# -- merge -----------------------------------------------------------------
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
			voc = liai.prep.Voc()
			test_sentences, _, _, _, _, Y_system_test, _, _, _, _, _ = liai.prep.df2ts(df_system_pred, voc, 0, 1)
			_, _, _, _, _, Y_lex_test, _, _, _, _, _ = liai.prep.df2ts(df_lex_pred, voc, 0, 1)

			for swap_a, swap_b in swaps:
				print(f'[{LANG} run {n}] Merging with {swap_a, swap_b}...')
				if swap_a == 'sys':
					sys_name = f'sys_TRAINDEV_{n}'
				elif swap_a == 'lex':
					sys_name = f'.lex_{TRAINDEV_CORPUS_NAME}'
				elif swap_a == 'rand':
					sys_name = f'rand'
				if swap_b == 'sys':
					lex_name = f'sys_TRAINDEV_{n}'
				elif swap_b == 'lex':
					lex_name = f'.lex_{TRAINDEV_CORPUS_NAME}'
				elif swap_b == 'rand':
					lex_name = f'rand'
				if swap_a == 'sent':
					sys_name = f'nosent.sys_TRAINDEV_{n}'
					lex_name = f'nosent.lex'
				
				LIAI_PRED_FILE = (
					f'{_blind_stem}'
					f'.liai_{_model_stem}'
					f'.{sys_name}'
					f'.{lex_name}'
					f'.dcupt'
				)
				LIAI_PRED_FILE_PATH = os.path.join(LIAI_PRED_PATH, LIAI_PRED_FILE)

				if os.path.exists(LIAI_PRED_FILE_PATH):
					print(f'[{LANG} run {n}] Loading cached LIAI predictions from {LIAI_PRED_FILE_PATH}')
					df_merged_pred = cupt_parser.setup_data_noTT(LIAI_PRED_FILE_PATH)
					merged_pred_mwes = cupt_parser.get_mwes(df_merged_pred)
				else:				
					if swap_a == 'sys':
						sys_data = Y_system_test
					elif swap_a == 'lex':
						sys_data = Y_lex_test
					elif swap_a == 'rand':
						sys_data = Y_system_test.reset_index().set_index(Y_system_test.reset_index()['sentence_id'].sample(frac=1, random_state=42))[0]
					if swap_b == 'sys':
						lex_data = Y_system_test
					elif swap_b == 'lex':
						lex_data = Y_lex_test
					elif swap_b == 'rand':
						lex_data = Y_lex_test.reset_index().set_index(Y_lex_test.reset_index()['sentence_id'].sample(frac=1, random_state=42))[0]
					if swap_a == 'sent':
						sys_data = Y_system_test
						lex_data = Y_system_test
						sentence_data = test_sentences.reset_index().set_index(test_sentences.reset_index()['sentence_id'].sample(frac=1, random_state=42))['form']
					else:
						sentence_data = test_sentences


					ts_pred = liai.predict(
						liai_model,
						test_sentences,
						sys_data,
						lex_data,
						device=DEVICE,
						batch_size=30
					)
					merged_pred_mwes = liai.prep.ts2df(
						df_test,
						cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes),
						ts_pred
					)
					print(f'[{LANG} run {n}] Saving LIAI predictions to {LIAI_PRED_FILE_PATH}')
					cupt_parser.write_matches_as_dcupt(merged_pred_mwes, TEST_CORPUS_BLIND_PATH, LIAI_PRED_FILE_PATH)
				end_time = perf_counter()
				print(f'[{LANG} run {n}] Merged in {(end_time - t_start):.2f}s: {len(cupt_parser.inline_mwes(merged_pred_mwes))} predicted MWEs')

				# -- score -----------------------------------------------------------------
				# scores_merged_lex_only, _  = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_lex_not_by_system)
				# scores_merged_both,     _  = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_lex_and_system)
				# scores_merged_system_only, _ = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_system_not_by_lex)
				# _, scores_merged_neither   = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, seen_by_lex_or_system)

				scores_merged_all          = oa_indices.full_eval(test_mwes, merged_pred_mwes)

				# scores_system_lex_only, _  = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_lex_not_by_system)
				# scores_system_both,     _  = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_lex_and_system)
				# scores_system_system_only, _ = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_system_not_by_lex)
				# _, scores_system_neither   = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, seen_by_lex_or_system)
				scores_system_all          = oa_indices.full_eval(test_mwes, system_pred_mwes)

				group_scores = {
					# ('merged', 'lex_only'): scores_merged_lex_only,
					# ('merged', 'both'):     scores_merged_both,
					# ('merged', 'system_only'): scores_merged_system_only,
					# ('merged', 'neither'):  scores_merged_neither,
					('merged', 'all'):      scores_merged_all,
					# ('system', 'lex_only'): scores_system_lex_only,
					# ('system', 'both'):     scores_system_both,
					# ('system', 'system_only'): scores_system_system_only,
					# ('system', 'neither'):  scores_system_neither,
					('system', 'all'):      scores_system_all,
				}
				for (method, group), scores in group_scores.items():
					for metric in METRICS:
						records.append({
							'lang':   LANG,
							'run':    n,
							'method': method,
							'group':  '_'.join([swap_a, swap_b]),
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
import numpy as np
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

class FlagMeta(type):
	def __repr__(cls):
		return cls._symbol

class Flag(metaclass=FlagMeta):
	_symbol = "?"
	def __str__(self): return self._symbol
	def __repr__(self): return self._symbol


class REJECTED(Flag): _symbol = "✗"

class NOT_REJECTED(Flag): _symbol = "~"

class INCONCLUSIVE(Flag): _symbol = "?"

class POSITIVE(Flag): _symbol = "✓"

ALPHA = 0.05

res = pd.DataFrame(columns=['language', 'a', 'b', 'hypothesis', 'null_rejected', 'a is normal', 'b is normal'])

summary = []
for lang in ALL_LANGS:
	# print(f'\n── {lang} ──')
	metric = 'f'


	s_msl = get_sample(lang, 'merged', 'sys_lex', 'f').rename('merged_sys_lex')
	s_mss = get_sample(lang, 'merged', 'sys_sys', 'f').rename('merged_sys_sys')
	s_mls = get_sample(lang, 'merged', 'lex_sys', 'f').rename('merged_lex_sys')
	s_mll = get_sample(lang, 'merged', 'lex_lex', 'f').rename('merged_lex_lex')
	s_mrr = get_sample(lang, 'merged', 'rand_rand', 'f').rename('merged_rand_rand').fillna(0)
	s_ms = get_sample(lang, 'merged', 'sent_sent', 'f').rename('merged_sent_sent').fillna(0)

	s_s = get_sample(lang, 'system', 'sys_lex', 'f').rename('system')



	f_null_rejected = lambda p : pd.notna(p) and (p < ALPHA)
	f_null_not_rejected = lambda p : pd.notna(p) and (p >= ALPHA)


	def _normality_test(s: pd.Series):
		def statistic(x, axis):
			# Get only the `normaltest` statistic; ignore approximate p-value
			return normaltest(x, axis=axis, nan_policy='omit').statistic

		if s.std() == 0:
			return INCONCLUSIVE

		x = norm.rvs(loc=s.mean(), scale=s.std(), size=50)
		rvs = lambda size: norm.rvs(loc=s.mean(), scale=s.std(), size=size)

		res = monte_carlo_test(
			x,
			rvs,
			statistic,
			alternative='greater',
			vectorized=True
		)
		return res.pvalue

	import itertools as it


	samples = [s_msl, s_mls, s_mss, s_mll, s_mrr, s_ms, s_s]

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





	def validate_hypothesis(pos_criterion, neg_criterion):
		if pos_criterion == experimentations.REJECTED:
			return experimentations.POSITIVE
		if neg_criterion == experimentations.REJECTED:
			return experimentations.NEGATIVE
		return experimentations.INCONCLUSIVE


	summary.append([
		lang,
		'base',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'system') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'system') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0]
		)
	])

	summary.append([
		lang,
		'swap',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_lex_sys') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_lex_sys') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0],
		)
	])


	summary.append([
		lang,
		'double mtlb',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_sys_sys') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_sys_sys') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0],
		)
	])

	summary.append([
		lang,
		'double mtlb vs mtlb',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_sys') & 
				(df['b'] == 'system') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_sys') & 
				(df['b'] == 'system') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
		)
	])
		

	summary.append([
		lang,
		'rand',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_rand_rand') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_rand_rand') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0],
		)
	])		

	summary.append([
		lang,
		'sent',
		validate_hypothesis(
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_sent_sent') & 
				(df['hypothesis'] == 'null_a_le_b')
			]['null_rejected'].values[0],
			tmp_res.loc[lambda df: 
				(df['a'] == 'merged_sys_lex') & 
				(df['b'] == 'merged_sent_sent') & 
				(df['hypothesis'] == 'null_a_ge_b')
			]['null_rejected'].values[0],
		)
	])		

	# criterion = tmp_res.loc[lambda df: 
	# 	(df['a'] == 'merged_sys_lex') & 
	# 	(df['b'] == 'merged_sent_sent') & 
	# 	(df['hypothesis'] == 'null_a_le_b')
	# 	]['null_rejected'].values[0]
	# if criterion == experimentations.REJECTED:
	# 	summary.append([lang, 'sent', experimentations.POSITIVE])

df_summary = pd.DataFrame(summary, columns=['language', 'hypothesis', 'result'])
df_summary

# %%
# ── Full per-language summary (all groups, all metrics) ────────────────────────
print('\n\n' + '='*70)
print('FULL PER-LANGUAGE SUMMARY (mean ± std across runs)')
print('='*70)
df_lang_with_multi = pd.concat([
	df_lang,
	df_multi_lang.assign(lang='[multilingual]')
], ignore_index=True)
df_summary = df_lang_with_multi.copy()
df_summary['mean±std'] = df_summary.apply(lambda r: f'{fmt(r["mean"])}±{fmt(r["std"])}', axis=1)
df_pivot_full = df_summary.pivot_table(
	index=['lang', 'metric'],
	columns=['method', 'group'],
	values='mean±std',
	aggfunc='first'
)
print(df_pivot_full.to_string())

# %%
df_scores_all.to_csv(os.path.join('liai_swap_scores_all.csv'), index=False)

# %%%
df_scores_all = pd.read_csv(os.path.join('liai_swap_scores_all.csv'))
# %%



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
