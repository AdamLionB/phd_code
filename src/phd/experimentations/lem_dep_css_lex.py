# from a import data2score_pipeline
# %%
from phd import lexicon, cupt_parser, lcss_lex, utils, oa_indices, lambdacss, scripts, liai
import torch
import pandas as pd
import os
from time import perf_counter
from collections import Counter

# %%


res = []
PARSEME_PATH = '../data/parseme'
PARSEME_VERSION = '1.2'
LANG = 'FR'
TRAIN_CORPUS_NAME = 'traindev'
TRAIN_CORPUS_PATH = f'{PARSEME_PATH}/{PARSEME_VERSION}/{LANG}/{TRAIN_CORPUS_NAME}.gold.dcupt'

lex_base = lambdacss.LambdaCSS_spec({'lemma' : True, 'deprel': False})

print(f'Loading "train" corpus from {TRAIN_CORPUS_PATH}')
t_start = perf_counter()
TT_train_lex, df_train_lex = cupt_parser.setup_data(
	TRAIN_CORPUS_PATH
)
t_end = perf_counter()
print(f'Corpus loaded in {(t_end - t_start):.2f}s: {len(TT_train_lex)} sentences, {len(df_train_lex)} tokens')


LEX_PATH = f'../data/lexicons'
LEX_JSON_NAME = f'tmp_{LANG}_{"".join(PARSEME_VERSION.split("."))}_{TRAIN_CORPUS_NAME}.json'
LEX_JSON_PATH = f'{LEX_PATH}/{LEX_JSON_NAME}'

print('Generating or loading lexicon from "train" corpus...')
lex_train = scripts.extract_lem_dep_css_lex.generate_or_load_lexicon_from_corpus_TT(TT_train_lex, LEX_JSON_PATH)
mwe_type_dist_train = Counter(lcss.get_mwe_type() for lcss in lex_train)
print(f'Lexicon generated or loaded: {mwe_type_dist_train.total()} entries, {len(mwe_type_dist_train)} unique MWE types, most common MWE type appears {mwe_type_dist_train.most_common(1)[0][1]} times')


# %% -----------------------------------------------------------------------------------------------
TEST_CORPUS_NAME = 'test'
TEST_CORPUS_GOLD_PATH = f'{PARSEME_PATH}/{PARSEME_VERSION}/{LANG}/{TEST_CORPUS_NAME}.gold.dcupt'
TEST_CORPUS_BLIND_PATH = f'{PARSEME_PATH}/{PARSEME_VERSION}/{LANG}/{TEST_CORPUS_NAME}.blind.cupt'


print(f'Loading "test" corpus from {TEST_CORPUS_GOLD_PATH}')
t_start = perf_counter()
TT_test, df_test = cupt_parser.setup_data(TEST_CORPUS_GOLD_PATH)
t_end = perf_counter()
print(f'Corpus loaded in {(t_end - t_start):.2f}s: {len(TT_test)} sentences, {len(df_test)} tokens')


print(f'Extracting mwe from test')
t_start = perf_counter()
test_mwes = cupt_parser.get_mwes(df_test)

test_inline_mwes = cupt_parser.inline_mwes(test_mwes)
t_end = perf_counter()
print(f'MWEs extracted in {(t_end - t_start):.2f}s: {len(test_inline_mwes)} MWEs')


# %% -----------------------------------------------------------------------------------------------
PRED_PATH = '../data/lex_pred'
PRED_FILE_PATH = f'{PARSEME_VERSION}/{LANG}/{TEST_CORPUS_NAME}.lex_lem_dep_css_trained_on_{TRAIN_CORPUS_NAME}.dcupt'

if os.path.exists(f'{PRED_PATH}/{PRED_FILE_PATH}'):
	print(f'Matching prediction from lexicon from file {PRED_PATH}/{PRED_FILE_PATH}')
	t_start = perf_counter()
	_, df_lex_pred = cupt_parser.setup_data(
		f'{PRED_PATH}/{PRED_FILE_PATH}'
	)
	lex_pred_mwes = cupt_parser.get_mwes(df_lex_pred)
	lex_pred_inline_mwes = cupt_parser.inline_mwes(lex_pred_mwes)
	t_end = perf_counter()
	print(f'Predictions loaded in {(t_end - t_start):.2f}s: {len(lex_pred_inline_mwes)} predicted MWEs')
else:

	print(f'Matching lexicon against test corpus')
	t_start = perf_counter()
	lex_pred_mwes = lex_train.match(df_test)
	lex_pred_inline_mwes = cupt_parser.inline_mwes(lex_pred_mwes)
	cupt_parser.write_matches_as_dcupt(lex_pred_mwes, TEST_CORPUS_BLIND_PATH, f'{PRED_PATH}/{PRED_FILE_PATH}')
	t_end = perf_counter()
	print(f'Lexicon matched against test corpus in {(t_end - t_start):.2f}s: {len(lex_pred_inline_mwes)} predicted MWEs')

#%%
SYSTEM_PATH = f'{PARSEME_PATH}/{PARSEME_VERSION}/system-results/MTLB-STRUCT.open/{LANG}/{TEST_CORPUS_NAME}.system.cupt'
print(f'Loading system predictions from {SYSTEM_PATH}')
t_start = perf_counter()
_, df_system_pred = cupt_parser.setup_data(
	SYSTEM_PATH
)
system_pred_mwes = cupt_parser.get_mwes(df_system_pred)
system_pred_inline_mwes = cupt_parser.inline_mwes(system_pred_mwes)
t_end = perf_counter()
print(f'System predictions loaded in {(t_end - t_start):.2f}s: {len(system_pred_inline_mwes)} predicted MWEs')
# %% -----------------------------------------------------------------------------------------------

print('merging predictions from lex and system')
t_start = perf_counter()
METHOD = 'LIAI'

if METHOD == 'union':
	merged_pred_mwes = cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes)
if METHOD == 'intersection':
	merged_pred_mwes = cupt_parser.intersection_mwes(lex_pred_mwes, system_pred_mwes)
if METHOD == 'LIAI':
	LIAI_PATH = '../data/liai'
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	FROZEN = False
	print('Loading LIAI model')
	liai_model = liai.Merger_with_padding_mask(4, FROZEN, DEVICE).to(DEVICE)
	state_dict = torch.load(
		f'{LIAI_PATH}/45_merger_masked.model',
		map_location=torch.device('cpu') if DEVICE == 'cpu' else None
	)
	state_dict.pop('bert.embedding.embeddings.position_ids', None)
	liai_model.load_state_dict(state_dict)

	voc = liai.prep.Voc()

	print('Preparing data for LIAI')
	t_start = perf_counter()
	test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = liai.prep.df2ts(df_system_pred, voc, 0, 1)
	_,_,_,_,_,Y_lex_test,_,_,_,_,_= liai.prep.df2ts(df_lex_pred, voc, 0, 1)
	# _,_,_,_,_,Y_truth_test,_,_,_,_,_ = liai.prep.df2ts(df_gold, voc, 0, 1)
	t_end = perf_counter()
	print(f'LIAI ready, data prepared in {(perf_counter() - t_start):.2f}s')


#%%
if METHOD == 'LIAI':
	print('Predicting with LIAI')
	liai_res = liai.predict(liai_model, test_sentences, Y_system_test, Y_lex_test, device=DEVICE, batch_size=1)


	# df2ts remaps sentence_ids; reconstruct the reverse map (same Python session => same set() ordering)
	_liai_rev_map = dict(enumerate(list(set(df_system_pred.reset_index()['sentence_id']))))

	# Build (original_sentence_id, frozenset_of_1indexed_token_ids) for every MWE in liai_res
	_liai_mwe_sigs = set()
	for _, row in liai_res.iterrows():
		orig_sid = _liai_rev_map[row['sentence_id']]
		tokens_0 = frozenset(i + 1 for i, v in enumerate(row[0]) if v == 1)
		tokens_m = frozenset(i + 1 for i, v in enumerate(row['_merge']) if v == 1)
		if tokens_0:
			_liai_mwe_sigs.add((orig_sid, tokens_0))
		if tokens_m:
			_liai_mwe_sigs.add((orig_sid, tokens_m))
	
	# Filter union: keep only MWEs whose (sentence_id, token set) appears in liai_res
	merged_pred_mwes = cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes).groupby(level=[0, 2]).filter(
		lambda g: (
			g.index.get_level_values('sentence_id')[0],
			frozenset(g['id'])
		) in _liai_mwe_sigs
	)

#%%
merged_pred_inline_mwes = cupt_parser.inline_mwes(merged_pred_mwes)
t_end = perf_counter()
print(f'Predictions merged in {(t_end - t_start):.2f}s: {len(merged_pred_inline_mwes)} predicted MWEs')


# %%
scores_merged_seen, scores_merged_unseen = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, mwe_type_dist_train.keys())
scores_merged_all = oa_indices.full_eval(test_mwes, merged_pred_mwes)

scores_system_seen, scores_system_unseen = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, mwe_type_dist_train.keys())
scores_system_all = oa_indices.full_eval(test_mwes, system_pred_mwes)


fmt = lambda x: f'{x:.4f}' if not pd.isna(x) else 'nan'

for metric in scores_merged_all.keys():
	# print(scores_merged_seen[metric])
	# print(scores_merged_unseen[metric])
	# print(scores_merged_all[metric])
	# print(scores_system_seen[metric])
	# print(scores_system_unseen[metric])
	# print(scores_system_all[metric])
	# print('---')
	print(f'{metric}: merged seen {fmt(scores_merged_seen[metric])}, merged unseen {fmt(scores_merged_unseen[metric])}, merged all {fmt(scores_merged_all[metric])} | system seen {fmt(scores_system_seen[metric])}, system unseen {fmt(scores_system_unseen[metric])}, system all {fmt(scores_system_all[metric])}')


# %%
