# %%
"""Richness-cap experiment.

For each PARSEME language, for each lexicon spec, for each richness cap K:
1. Generate / load a richness-capped lexicon  (entries selected by MWE type)
2. Match the capped lexicon against the test data
3. Write the predictions as a .dcupt file
4. Evaluate lexicon-only predictions  (P / R / F + diversity indices)
5. (TODO) Run LIAI merger once mtlb system predictions are available

All heavy artefacts (lexicon JSONs, annotation .dcupt files) are cached on
disk and only regenerated when missing.
"""

from phd import lexicon, cupt_parser, lcss_lex, lambdacss, utils, oa_indices
from phd.scripts.generate_richness_capped_lexicons import (
	lcss_adapter_factory,
	seqrep_adapter_factory,
	get_or_create_full_lexicon,
	get_or_create_capped_lexicon,
	get_or_create_obs_capped_lexicon,
	get_observed_types,
	compute_richness,
)
import pandas as pd
import os
import rich.console

console = rich.console.Console()

# %%
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARSEME_PATH = '../data/parseme'
PARSEME_VERSION = '1.2'
LEXICON_DIR = '../data/lexicons/richness_cap'
RESULTS_DIR = '../data/results/richness_cap'

K_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]
SEED = 42

SKIP_DIRS = {
	'RO', 'EU', 'Multi', 'Multi_no_EU_RO',
	'bin', 'trial', 'system-results',
}

SPECS = {
	'lem_dep_css': lcss_adapter_factory(
		lambdacss.LambdaCSS_spec({'lemma': True, 'deprel': False})
	),
	'seqrep_lem': seqrep_adapter_factory(
		lexicon.Seq_rep_spec(('lemma',))
	),
}


# %%
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_languages():
	version_dir = f'{PARSEME_PATH}/{PARSEME_VERSION}'
	return sorted([
		d for d in os.listdir(version_dir)
		if os.path.isdir(os.path.join(version_dir, d))
		and d not in SKIP_DIRS
	])


def match_lexicon(lex, df_test):
	"""Match a lexicon against test data, dispatching by adapter type."""
	if isinstance(lex.entry_type, lcss_lex.LCSSAdapter):
		return lex.match(df_test)
	elif isinstance(lex.entry_type, lexicon.SeqRepAdapter):
		sid = utils.sort_column(df_test)
		return lex.match(df_test, sid)
	else:
		raise TypeError(f"Unknown adapter: {type(lex.entry_type)}")


def evaluate_and_write(lex, lang, spec_name, k_label, df_test, truth, test_blind_path):
	"""Match, write dcupt, evaluate.  Returns metrics dict."""
	lang_dir = f'{PARSEME_PATH}/{PARSEME_VERSION}/{lang}'
	dcupt_path = f'{lang_dir}/test.lex_{spec_name}_{k_label}.dcupt'

	pred = match_lexicon(lex, df_test)

	if not os.path.exists(dcupt_path):
		cupt_parser.write_matches_as_dcupt(pred, test_blind_path, dcupt_path)

	if truth.empty or pred.empty:
		return {'p': 0, 'r': 0, 'f': 0}

	try:
		return oa_indices.full_eval(truth, pred)
	except Exception:
		inline_truth, inline_pred, inline_tp = oa_indices.true_pred(truth, pred)
		p, r, f = oa_indices.P_R_F(inline_truth, inline_pred, inline_tp)
		return {'p': p, 'r': r, 'f': f}


# %%
# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

results = []
# langs = get_languages()
langs = ['FR']
console.print(f'Languages: {langs}')

for lang in langs:
	console.print(f'\n[bold]=== {lang} ===[/bold]')

	lang_dir = f'{PARSEME_PATH}/{PARSEME_VERSION}/{lang}'
	traindev_path = f'{lang_dir}/traindev.gold.dcupt'
	test_gold_path = f'{lang_dir}/test.gold.dcupt'
	test_blind_path = f'{lang_dir}/test.blind.cupt'

	if not os.path.exists(traindev_path):
		console.print('  [yellow]traindev.gold.dcupt not found, skipping[/yellow]')
		continue
	if not os.path.exists(test_gold_path):
		console.print('  [yellow]test.gold.dcupt not found, skipping[/yellow]')
		continue

	# Parse test data once per language (expensive)
	with console.status(f'Parsing test data for {lang}...'):
		TT_test, df_test = cupt_parser.setup_data(test_gold_path)
		truth = cupt_parser.get_mwes(df_test)

	for spec_name, adapter_factory in SPECS.items():
		console.print(f'  [cyan]Spec: {spec_name}[/cyan]')

		# Full lexicon
		with console.status(f'Loading full lexicon {lang}/{spec_name}...'):
			lex, _ = get_or_create_full_lexicon(
				lang, spec_name, adapter_factory, traindev_path, LEXICON_DIR
			)

		richness = compute_richness(lex)
		console.print(f'    Richness: {richness} types, {len(lex)} entries')

		# Evaluate full lexicon
		with console.status(f'Evaluating full {lang}/{spec_name}...'):
			full_eval = evaluate_and_write(
				lex, lang, spec_name, 'full', df_test, truth, test_blind_path
			)

		results.append({
			'lang': lang,
			'spec': spec_name,
			'cap': 'none',
			'K': 'full',
			'richness': richness,
			'n_entries': len(lex),
			**full_eval,
		})
		console.print(
			f'    Full: P={full_eval.get("p", 0):.3f} '
			f'R={full_eval.get("r", 0):.3f} '
			f'F={full_eval.get("f", 0):.3f}'
		)

		# Innate-capped lexicons
		for k in K_VALUES:
			with console.status(f'innate K={k} for {lang}/{spec_name}...'):
				capped, _ = get_or_create_capped_lexicon(
					lex, lang, spec_name, k, LEXICON_DIR, seed=SEED
				)

			if capped is None:
				continue

			capped_richness = compute_richness(capped)

			with console.status(f'Evaluating innate K={k} for {lang}/{spec_name}...'):
				cap_eval = evaluate_and_write(
					capped, lang, spec_name, f'K{k}',
					df_test, truth, test_blind_path,
				)

			results.append({
				'lang': lang,
				'spec': spec_name,
				'cap': 'innate',
				'K': k,
				'richness': capped_richness,
				'n_entries': len(capped),
				**cap_eval,
			})
			console.print(
				f'    innate K={k}: P={cap_eval.get("p", 0):.3f} '
				f'R={cap_eval.get("r", 0):.3f} '
				f'F={cap_eval.get("f", 0):.3f} '
				f'(entries={len(capped)})'
			)

		# Observational-capped lexicons
		with console.status(f'Matching full lexicon for obs richness {lang}/{spec_name}...'):
			full_pred = match_lexicon(lex, df_test)
			observed = get_observed_types(full_pred)

		obs_richness = len(observed)
		console.print(f'    Observational richness: {obs_richness} types')

		for k in K_VALUES:
			with console.status(f'obs K={k} for {lang}/{spec_name}...'):
				capped, _ = get_or_create_obs_capped_lexicon(
					lex, observed, lang, spec_name, k, LEXICON_DIR, seed=SEED
				)

			if capped is None:
				continue

			capped_richness = compute_richness(capped)

			with console.status(f'Evaluating obs K={k} for {lang}/{spec_name}...'):
				cap_eval = evaluate_and_write(
					capped, lang, spec_name, f'obsK{k}',
					df_test, truth, test_blind_path,
				)

			results.append({
				'lang': lang,
				'spec': spec_name,
				'cap': 'observational',
				'K': k,
				'richness': capped_richness,
				'n_entries': len(capped),
				**cap_eval,
			})
			console.print(
				f'    obs K={k}: P={cap_eval.get("p", 0):.3f} '
				f'R={cap_eval.get("r", 0):.3f} '
				f'F={cap_eval.get("f", 0):.3f} '
				f'(entries={len(capped)})'
			)

# %%
# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

df_results = pd.DataFrame.from_records(results)
console.print('\n[bold]Results table[/bold]')
console.print(df_results.to_string())

# %%
# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

# Detail: all columns, one row per (lang, spec, cap, K)
detail_path = os.path.join(RESULTS_DIR, 'detail.csv')
df_results.to_csv(detail_path, index=False)
console.print(f'\nDetail saved to [green]{detail_path}[/green]')

# Recap: key metrics only
recap_cols = ['lang', 'spec', 'cap', 'K', 'richness', 'n_entries', 'p', 'r', 'f']
recap_cols = [c for c in recap_cols if c in df_results.columns]
df_recap = df_results[recap_cols].copy()
recap_path = os.path.join(RESULTS_DIR, 'recap.csv')
df_recap.to_csv(recap_path, index=False)
console.print(f'Recap saved to [green]{recap_path}[/green]')

# %%
# ---------------------------------------------------------------------------
# LIAI evaluation
# ---------------------------------------------------------------------------
# Requires mtlb system predictions at
#   {lang_dir}/test.mtlb_trained_on_train.cupt

from phd import liai
import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
liai_path = '../data/liai'

liai_results = []

for lang in langs:
	lang_dir = f'{PARSEME_PATH}/{PARSEME_VERSION}/{lang}'
	system_path = f'{lang_dir}/test.mtlb_trained_on_train.cupt'
	if not os.path.exists(system_path):
		console.print(f'[yellow]{lang}: no mtlb system predictions, skipping LIAI[/yellow]')
		continue

	console.print(f'\n[bold]=== LIAI {lang} ===[/bold]')

	voc = liai.prep.Voc()
	with console.status(f'Loading system predictions for {lang}...'):
		test_sentences, Y_system = (
			t := liai.prep.file2ts(system_path, voc, 0, 1)
		)[0], t[5]
		Y_truth = liai.prep.file2ts(f'{lang_dir}/test.cupt', voc, 0, 1)[5]

	# bert_cache = str(Path(
	#     '~/.cache/huggingface/hub/models--bert-base-multilingual-cased/'
	#     'snapshots/fdfce55e83dbed325647a63e7e1f5de19f0382ba'
	# ).expanduser().resolve())
	bert_cache = None

	with console.status(f'Loading LIAI model for {lang}...'):
		model = liai.Merger_with_padding_mask(
			4, False, device, bert_custom_cache=bert_cache
		).to(device)
		state_dict = torch.load(
			f'{liai_path}/45_merger_masked.model', map_location=device
		)
		state_dict.pop('bert.embedding.embeddings.position_ids', None)
		model.load_state_dict(state_dict)

	df_test_liai = cupt_parser.setup_data_noTT(f'{lang_dir}/test.cupt')

	k_labels = (
		[('none', 'full')]
		+ [('innate', f'K{k}') for k in K_VALUES]
		+ [('observational', f'obsK{k}') for k in K_VALUES]
	)

	for spec_name in SPECS:
		for cap_type, k_label in k_labels:
			lex_dcupt = f'{lang_dir}/test.lex_{spec_name}_{k_label}.dcupt'
			if not os.path.exists(lex_dcupt):
				continue

			console.print(f'  {spec_name} / {k_label}')
			with console.status(f'LIAI eval {lang}/{spec_name}/{k_label}...'):
				Y_lex = liai.prep.file2ts(lex_dcupt, voc, 0, 1)[5]
				truth_data, test_data = liai.build_candidate_table_and_labels(
					Y_system, Y_lex, Y_truth
				)
				eval_res = liai.eval_model(
					model, test_sentences, Y_truth,
					test_data, truth_data, df_test_liai, device, 100
				)

			k_val = k_label if k_label == 'full' else k_label.replace('K', '').replace('obs', '')
			liai_results.append({
				'lang': lang,
				'spec': spec_name,
				'cap': cap_type,
				'K': k_val,
				**eval_res,
			})
			console.print(
				f'    P={eval_res["precision"]:.3f} '
				f'R={eval_res["recall"]:.3f} '
				f'F={eval_res["f1"]:.3f}'
			)

# %%
# ---------------------------------------------------------------------------
# Save LIAI results
# ---------------------------------------------------------------------------

if liai_results:
	df_liai = pd.DataFrame.from_records(liai_results)
	console.print('\n[bold]LIAI Results table[/bold]')
	console.print(df_liai.to_string())

	os.makedirs(RESULTS_DIR, exist_ok=True)

	liai_detail_path = os.path.join(RESULTS_DIR, 'liai_detail.csv')
	df_liai.to_csv(liai_detail_path, index=False)
	console.print(f'\nLIAI detail saved to [green]{liai_detail_path}[/green]')

	liai_recap_cols = ['lang', 'spec', 'cap', 'K', 'precision', 'recall', 'f1', 'richness', 'E_1mD', 'expS']
	liai_recap_cols = [c for c in liai_recap_cols if c in df_liai.columns]
	df_liai_recap = df_liai[liai_recap_cols].copy()
	liai_recap_path = os.path.join(RESULTS_DIR, 'liai_recap.csv')
	df_liai_recap.to_csv(liai_recap_path, index=False)
	console.print(f'LIAI recap saved to [green]{liai_recap_path}[/green]')

# %%
