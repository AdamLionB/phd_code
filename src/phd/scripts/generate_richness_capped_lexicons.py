"""Generate full and richness-capped lexicons for all PARSEME languages.

For each language and lexicon spec, this script:
1. Generates a full lexicon from traindev data (if not already on disk)
2. Computes innate richness = number of unique MWE types (by fmset of lemmas)
3. For each richness cap value K:
   - Randomly selects K MWE types (seed-based)
   - Keeps all entries whose type is among the selected K
   - Saves capped lexicon as JSON (if not already on disk)
"""

import os
import sys
import random
from collections import defaultdict

from phd import lexicon, cupt_parser, lcss_lex, lambdacss, utils


# ---------------------------------------------------------------------------
# MWE type extraction
# ---------------------------------------------------------------------------

def get_entry_mwe_type(entry):
	"""Get the MWE type (fmset of lemmas) from a lexicon entry."""
	if isinstance(entry, lambdacss.LambdaCSS):
		return entry.expr()
	elif isinstance(entry, lexicon.Seq_rep):
		return utils.fmset([c['lemma'] for c in entry.components])
	else:
		raise TypeError(f"Unknown entry type: {type(entry)}")


def compute_richness(lex):
	"""Compute the number of unique MWE types in a lexicon."""
	return len({get_entry_mwe_type(entry) for entry in lex})


# ---------------------------------------------------------------------------
# Adapter factories
# ---------------------------------------------------------------------------

def lcss_adapter_factory(spec):
	"""Create a factory for LambdaCSS lexicon adapters.

	The factory callable takes a traindev path and returns (adapter, data)
	where data is suitable for adapter.instantiate().
	"""
	def factory(traindev_path):
		TTs, _ = cupt_parser.setup_data(traindev_path)
		return lcss_lex.LCSSAdapter(spec), TTs
	return factory


def seqrep_adapter_factory(spec):
	"""Create a factory for SeqRep lexicon adapters."""
	def factory(traindev_path):
		_, df = cupt_parser.setup_data(traindev_path)
		return lexicon.SeqRepAdapter(spec), df
	return factory


# ---------------------------------------------------------------------------
# Lexicon generation & capping
# ---------------------------------------------------------------------------

def get_or_create_full_lexicon(lang, spec_name, adapter_factory, traindev_path, output_dir):
	"""Get or create the full (uncapped) lexicon for a language and spec.

	Returns (MWE_lexicon, json_path).
	"""
	json_path = os.path.join(output_dir, f'{lang}.{spec_name}.json')
	if os.path.exists(json_path):
		return lexicon.MWE_lexicon.from_json(json_path), json_path

	entry_type, data = adapter_factory(traindev_path)
	lex = lexicon.MWE_lexicon(entry_type.instantiate(data), entry_type)

	os.makedirs(output_dir, exist_ok=True)
	lex.to_json(json_path)
	return lex, json_path


def cap_lexicon(lex, k, seed=42):
	"""Cap a lexicon to K MWE types (by fmset of lemmas).

	Randomly selects K types and keeps ALL entries whose type is among
	the selected K.  Returns None if K >= number of unique types.
	"""
	type_to_entries = defaultdict(list)
	for entry in lex:
		t = get_entry_mwe_type(entry)
		type_to_entries[t].append(entry)

	all_types = sorted(type_to_entries.keys())
	if k >= len(all_types):
		return None

	rng = random.Random(seed)
	selected_types = rng.sample(all_types, k)

	capped_entries = [
		entry
		for t in selected_types
		for entry in type_to_entries[t]
	]
	return lexicon.MWE_lexicon(capped_entries, lex.entry_type)


def get_or_create_capped_lexicon(lex, lang, spec_name, k, output_dir, seed=42):
	"""Get or create a capped lexicon for a given K.

	Returns (MWE_lexicon | None, json_path | None).
	None when K >= number of unique types (skipped).
	"""
	json_path = os.path.join(output_dir, f'{lang}.{spec_name}.K{k}.json')
	if os.path.exists(json_path):
		return lexicon.MWE_lexicon.from_json(json_path), json_path

	capped = cap_lexicon(lex, k, seed=seed)
	if capped is None:
		return None, None

	os.makedirs(output_dir, exist_ok=True)
	capped.to_json(json_path)
	return capped, json_path


# ---------------------------------------------------------------------------
# Observational richness capping
# ---------------------------------------------------------------------------

def get_observed_types(matches):
	"""Extract the set of observed MWE types from match results.

	Parameters
	----------
	matches : DataFrame
		Match results with (sentence_id, token_id, mwe_id) multi-index
		and a 'lemma' column.

	Returns
	-------
	set[fmset]
		Unique MWE types that were actually matched.
	"""
	if matches.empty:
		return set()
	return {
		utils.fmset(list(group['lemma']))
		for _, group in matches.groupby(level=[0, 2])
	}


def cap_lexicon_observational(lex, observed_types, k, seed=42):
	"""Cap a lexicon to K *observed* MWE types.

	Only lexicon entries whose type was actually observed in test predictions
	are candidates for selection.  Randomly selects K observed types and
	keeps ALL entries of those types.

	Returns None if K >= number of observed types.
	"""
	observed_types_sorted = sorted(observed_types)
	if k >= len(observed_types_sorted):
		return None

	type_to_entries = defaultdict(list)
	for entry in lex:
		t = get_entry_mwe_type(entry)
		if t in observed_types:
			type_to_entries[t].append(entry)

	# Only sample from types that have at least one matching entry
	selectable = sorted(type_to_entries.keys())
	if k >= len(selectable):
		return None

	rng = random.Random(seed)
	selected_types = rng.sample(selectable, k)

	capped_entries = [
		entry
		for t in selected_types
		for entry in type_to_entries[t]
	]
	return lexicon.MWE_lexicon(capped_entries, lex.entry_type)


def get_or_create_obs_capped_lexicon(
	lex, observed_types, lang, spec_name, k, output_dir, seed=42
):
	"""Get or create an observationally-capped lexicon for a given K.

	Returns (MWE_lexicon | None, json_path | None).
	None when K >= number of observed types (skipped).
	"""
	json_path = os.path.join(output_dir, f'{lang}.{spec_name}.obsK{k}.json')
	if os.path.exists(json_path):
		return lexicon.MWE_lexicon.from_json(json_path), json_path

	capped = cap_lexicon_observational(lex, observed_types, k, seed=seed)
	if capped is None:
		return None, None

	os.makedirs(output_dir, exist_ok=True)
	capped.to_json(json_path)
	return capped, json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
	parseme_data_dir,
	output_dir,
	specs,
	k_values=(16, 32, 64, 128, 256, 512),
	seed=42,
	skip_dirs=None,
):
	"""Generate full and capped lexicons for all languages.

	Parameters
	----------
	parseme_data_dir : str
		Path to a parseme version directory (e.g. .../parseme/1.2).
	output_dir : str
		Where to store the lexicon JSON files.
	specs : dict[str, callable]
		Mapping of spec_name → adapter_factory (from lcss_adapter_factory
		or seqrep_adapter_factory).
	k_values : tuple[int, ...]
		Richness cap values.
	seed : int
		Random seed for reproducible capping.
	skip_dirs : set[str] | None
		Directory names to skip inside parseme_data_dir.
	"""
	if skip_dirs is None:
		skip_dirs = set()

	langs = sorted([
		d for d in os.listdir(parseme_data_dir)
		if os.path.isdir(os.path.join(parseme_data_dir, d))
		and d not in skip_dirs
	])

	results = {}

	for lang in langs:
		print(f'\n=== {lang} ===')
		traindev_path = os.path.join(parseme_data_dir, lang, 'traindev.gold.dcupt')
		if not os.path.exists(traindev_path):
			print(f'  WARNING: {traindev_path} not found, skipping.')
			continue

		for spec_name, adapter_factory in specs.items():
			print(f'  Spec: {spec_name}')
			lex, _ = get_or_create_full_lexicon(
				lang, spec_name, adapter_factory, traindev_path, output_dir
			)

			richness = compute_richness(lex)
			print(f'    Richness (unique MWE types): {richness}, entries: {len(lex)}')
			results[(lang, spec_name)] = {
				'richness': richness,
				'n_entries': len(lex),
			}

			for k in k_values:
				capped, path = get_or_create_capped_lexicon(
					lex, lang, spec_name, k, output_dir, seed=seed
				)
				if capped is None:
					print(f'    K={k}: skipped (fewer types than K)')
				else:
					print(f'    K={k}: {len(capped)} entries')

	return results


if __name__ == '__main__':
	parseme_data_dir = os.path.join(
		os.path.dirname(__file__), '..', 'data', 'parseme', '1.2'
	)
	output_dir = os.path.join(
		os.path.dirname(__file__), '..', 'data', 'lexicons', 'richness_cap'
	)

	skip_dirs = {
		'RO', 'EU', 'Multi', 'Multi_no_EU_RO',
		'bin', 'trial', 'system-results',
	}

	specs = {
		'lem_dep_css': lcss_adapter_factory(
			lambdacss.LambdaCSS_spec({'lemma': True, 'deprel': False})
		),
		'seqrep_lem': seqrep_adapter_factory(
			lexicon.Seq_rep_spec(('lemma',))
		),
	}

	main(parseme_data_dir, output_dir, specs, skip_dirs=skip_dirs)
