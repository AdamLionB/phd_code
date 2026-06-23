# from a import data2score_pipeline
# %%
from phd import lexicon, cupt_parser, lcss_lex, utils, oa_indices, lambdacss, scripts, liai
import torch
import pandas as pd
import os
from time import perf_counter
from collections import Counter
from pathlib import Path
from scipy.stats import norm, f_oneway, ttest_rel, pearsonr

def load_or_compute_and_save_prediction(target_path, base_path, predict_func):
	if os.path.exists(target_path):
		df_pred = cupt_parser.setup_data_noTT(
			target_path
		)
		return cupt_parser.get_mwes(df_pred)
	else:
		mwes_pred = predict_func()
		cupt_parser.write_matches_as_dcupt(mwes_pred, base_path, target_path)
		return mwes_pred


# %%


res = []
PARSEME_PATH = os.path.join('..', 'data', 'parseme')
PARSEME_VERSION = '1.2'
LANG = 'DE'
TRAIN_CORPUS_NAME = 'traindev'
TRAIN_CORPUS_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TRAIN_CORPUS_NAME}.gold.dcupt')

# LEX_TRAIN_CORPUS_NAME = 'traindev'
# LEX_TRAIN_CORPUS_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TRAIN_CORPUS_NAME}.gold.dcupt')

lex_base = lambdacss.LambdaCSS_spec({'lemma' : True, 'deprel': False})

print(f'Loading "train" corpus from {TRAIN_CORPUS_PATH}')
t_start = perf_counter()
TT_train_lex, df_train_lex = cupt_parser.setup_data(
	TRAIN_CORPUS_PATH
)
t_end = perf_counter()
print(f'Corpus loaded in {(t_end - t_start):.2f}s: {len(TT_train_lex)} sentences, {len(df_train_lex)} tokens')


LEX_PATH = os.path.join('..', 'data', 'lexicons')
LEX_JSON_NAME = f'tmp_{LANG}_{"".join(PARSEME_VERSION.split("."))}_{TRAIN_CORPUS_NAME}.json'
LEX_JSON_PATH = os.path.join(LEX_PATH, LEX_JSON_NAME)

print('Generating or loading lexicon from "train" corpus...')
lex_train = scripts.extract_lem_dep_css_lex.generate_or_load_lexicon_from_corpus_TT(TT_train_lex, LEX_JSON_PATH)
mwe_type_dist_train = Counter(lcss.get_mwe_type() for lcss in lex_train)
print(f'Lexicon generated or loaded: {sum(mwe_type_dist_train.values())} entries, {len(mwe_type_dist_train)} unique MWE types, most common MWE type appears {mwe_type_dist_train.most_common(1)[0][1]} times')


# %% -----------------------------------------------------------------------------------------------
TEST_CORPUS_NAME = 'test'
TEST_CORPUS_GOLD_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.gold.dcupt')
TEST_CORPUS_BLIND_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.blind.cupt')


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
PRED_PATH = os.path.join('..', 'data', 'lex_pred')
LEX_PRED_NAME = f'{TEST_CORPUS_NAME}.lex_lem_dep_css_trained_on_{TRAIN_CORPUS_NAME}.dcupt'
PRED_FILE_PATH = os.path.join(PRED_PATH, PARSEME_VERSION, LANG, LEX_PRED_NAME)

if os.path.exists(PRED_FILE_PATH):
	print(f'Matching prediction from lexicon from file PRED_FILE_PATH')
	t_start = perf_counter()
	df_lex_pred = cupt_parser.setup_data_noTT(
		PRED_FILE_PATH
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
	cupt_parser.write_matches_as_dcupt(lex_pred_mwes, TEST_CORPUS_BLIND_PATH, PRED_FILE_PATH)
	df_lex_pred = cupt_parser.setup_data_noTT(
		PRED_FILE_PATH
	)
	t_end = perf_counter()
	print(f'Lexicon matched against test corpus in {(t_end - t_start):.2f}s: {len(lex_pred_inline_mwes)} predicted MWEs')

#%%
SYSTEM_PATH = os.path.join(PARSEME_PATH, PARSEME_VERSION, 'system-results', 'MTLB-STRUCT.open', LANG, f'{TEST_CORPUS_NAME}.system.cupt')

MTLB_PRED_PATH = os.path.join('..', 'data', 'mtlb_pred')

N = '10'
s_seen_merged = []
s_unseen_merged = []
s_merged = []
s_seen_system = []
s_unseen_system = []
s_system = []
for n in range(1, int(N)+1):

	SYSTEM_PATH = os.path.join(MTLB_PRED_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.mtlb_trained_on_TRAINDEV_{n}.dcupt')

	# SYSTEM_PATH = os.path.join('..', '..', '..', '..', 'MTLB-STRUCT', 'code', 'saved', 'DE_TEST_bert-base-german-cased_single', 'test.mtlb_trained_on_TRAINDEV.cupt')
	# SYSTEM_PATH = os.path.join('..', '..', '..', '..', 'MTLB-STRUCT', 'code', 'saved', 'DE_TEST_dbmdz-bert-base-german-uncased_single', 'test.mtlb_trained_on_TRAINDEV.cupt')
	# SYSTEM_PATH = os.path.join('..', '..', '..', '..', 'MTLB-STRUCT', 'code', 'saved', 'DE_TEST_dbmdz-bert-base-german-uncased_single', 'test.mtlb_trained_on_TRAIN.cupt')

	print(f'Loading system predictions from {SYSTEM_PATH}')
	t_start = perf_counter()
	_, df_system_pred = cupt_parser.setup_data(
		SYSTEM_PATH
	)
	system_pred_mwes = cupt_parser.get_mwes(df_system_pred)
	system_pred_inline_mwes = cupt_parser.inline_mwes(system_pred_mwes)
	t_end = perf_counter()
	print(f'System predictions loaded in {(t_end - t_start):.2f}s: {len(system_pred_inline_mwes)} predicted MWEs')

	print('merging predictions from lex and system')
	t_start = perf_counter()
	METHOD = 'LIAI'

	if METHOD == 'union':
		merged_pred_mwes = cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes)
	if METHOD == 'intersection':
		merged_pred_mwes = cupt_parser.intersection_mwes(lex_pred_mwes, system_pred_mwes)
	if METHOD == 'LIAI':
		LIAI_MODEL_PATH = os.path.join('..', 'data', 'liai')
		LIAI_MODEL_NAME = '45_merger_masked.model'
		DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
		FROZEN = False
		print('Loading LIAI model')

		# BERT_CUSTOM_CACHE = str(Path('~/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/fdfce55e83dbed325647a63e7e1f5de19f0382ba').expanduser().resolve())
		BERT_CUSTOM_CACHE = None

		liai_model = liai.Merger_with_padding_mask(
			4,
			FROZEN,
			DEVICE,
			BERT_CUSTOM_CACHE
		).to(DEVICE)
		state_dict = torch.load(
			os.path.join(LIAI_MODEL_PATH, LIAI_MODEL_NAME),
			map_location=torch.device('cpu') if DEVICE == 'cpu' else None
		)

		try:
			# older versions
			liai_model.load_state_dict(state_dict)
		except:
			# newer versions
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

		LIAI_PRED_PATH = os.path.join('..', 'data', 'liai_pred')
		LIAI_PRED_FILE_PATH = os.path.join(LIAI_PRED_PATH, PARSEME_VERSION, LANG, f'{TEST_CORPUS_NAME}.{LIAI_MODEL_NAME.removesuffix(".model")}39_{LEX_PRED_NAME}')
		
		def predict_func():
			ts_pred = liai.predict(liai_model, test_sentences, Y_system_test, Y_lex_test, device=DEVICE, batch_size=30)
			return liai.prep.ts2df(
				df_test,
				cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes),
				ts_pred
				)
		print('Predicting with LIAI or loading predictions')
		merged_pred_mwes = predict_func()
	# merged_pred_mwes = load_or_compute_and_save_prediction(
	# 	LIAI_PRED_FILE_PATH,
	# 	TEST_CORPUS_BLIND_PATH,
	# 	predict_func
	# )
	

	# merged_pred_mwes = liai.prep.ts2df(
	# 	df_test,
	# 	cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes),
	# 	liai_res
	# 	)
	# # df2ts remaps sentence_ids; reconstruct the reverse map 
	# _liai_rev_map = dict(enumerate(list(set(df_system_pred.reset_index()['sentence_id']))))

	# # Build (original_sentence_id, frozenset_of_1indexed_token_ids) for every MWE in liai_res
	# _liai_mwe_sigs = set()
	# for _, row in liai_res.iterrows():
	# 	orig_sid = _liai_rev_map[row['sentence_id']]
	# 	tokens_0 = frozenset(i + 1 for i, v in enumerate(row[0]) if v == 1)
	# 	tokens_m = frozenset(i + 1 for i, v in enumerate(row['_merge']) if v == 1)
	# 	if tokens_0:
	# 		_liai_mwe_sigs.add((orig_sid, tokens_0))
	# 	if tokens_m:
	# 		_liai_mwe_sigs.add((orig_sid, tokens_m))
	
	# # Filter union: keep only MWEs whose (sentence_id, token set) appears in liai_res
	# merged_pred_mwes = cupt_parser.union_mwes(lex_pred_mwes, system_pred_mwes).groupby(level=[0, 2]).filter(
	# 	lambda g: (
	# 		g.index.get_level_values('sentence_id')[0],
	# 		frozenset(g['id'])
	# 	) in _liai_mwe_sigs
	# )

	merged_pred_inline_mwes = cupt_parser.inline_mwes(merged_pred_mwes)
	t_end = perf_counter()
	print(f'Predictions merged in {(t_end - t_start):.2f}s: {len(merged_pred_inline_mwes)} predicted MWEs')



	scores_merged_seen, scores_merged_unseen = oa_indices.seen_unseen_full_eval(test_mwes, merged_pred_mwes, mwe_type_dist_train.keys())
	scores_merged_all = oa_indices.full_eval(test_mwes, merged_pred_mwes)

	scores_system_seen, scores_system_unseen = oa_indices.seen_unseen_full_eval(test_mwes, system_pred_mwes, mwe_type_dist_train.keys())
	scores_system_all = oa_indices.full_eval(test_mwes, system_pred_mwes)


	fmt = lambda x: f'{x:.4f}' if not pd.isna(x) else 'nan'

	# for metric in scores_merged_all.keys():
	# 	print(f'{metric}: merged seen {fmt(scores_merged_seen[metric])}, merged unseen {fmt(scores_merged_unseen[metric])}, merged all {fmt(scores_merged_all[metric])} | system seen {fmt(scores_system_seen[metric])}, system unseen {fmt(scores_system_unseen[metric])}, system all {fmt(scores_system_all[metric])}')

	s_seen_merged.append(scores_merged_seen)
	s_unseen_merged.append(scores_merged_unseen)
	s_merged.append(scores_merged_all)

	s_seen_system.append(scores_system_seen)
	s_unseen_system.append(scores_system_unseen)
	s_system.append(scores_system_all)
# %%
metrics = s_merged[0].keys()

def avg(scores, metric):
	return sum(s[metric] for s in scores if not pd.isna(s[metric]))/len(scores)

def std(scores, metric):
	mean = avg(scores, metric)
	return (sum((s[metric] - mean)**2 for s in scores if not pd.isna(s[metric]))/len(scores))**0.5

def gaussian_sup(mu1, sigma1, mu2, sigma2):
	norm_diff = norm(loc=mu1 - mu2, scale=(sigma1**2 + sigma2**2) ** 0.5)
	return norm_diff.sf(0)

def anova_pvalue(scores1, scores2, metric):
	vals1 = [s[metric] for s in scores1 if not pd.isna(s[metric])]
	vals2 = [s[metric] for s in scores2 if not pd.isna(s[metric])]
	if len(vals1) < 2 or len(vals2) < 2:
		return float('nan')
	_, p = f_oneway(vals1, vals2)
	return p

def paired_ttest_pvalue(scores1, scores2, metric):
	"""Paired t-test: scores1[n] and scores2[n] come from the same run n."""
	pairs = [(s1[metric], s2[metric]) for s1, s2 in zip(scores1, scores2)
	         if not pd.isna(s1[metric]) and not pd.isna(s2[metric])]
	if len(pairs) < 2:
		return float('nan')
	vals1, vals2 = zip(*pairs)
	_, p = ttest_rel(vals1, vals2)
	return p

def pair_correlation(scores1, scores2, metric):
	"""Pearson r between paired score series — high r means pairing matters."""
	pairs = [(s1[metric], s2[metric]) for s1, s2 in zip(scores1, scores2)
	         if not pd.isna(s1[metric]) and not pd.isna(s2[metric])]
	if len(pairs) < 2:
		return float('nan')
	vals1, vals2 = zip(*pairs)
	r, _ = pearsonr(vals1, vals2)
	return r

for metric in metrics:
	print(
		f'{metric} avg: merged seen {fmt(avg(s_seen_merged, metric))},', 
		f'merged unseen {fmt(avg(s_unseen_merged, metric))},',
		f'merged all {fmt(avg(s_merged, metric))} |', 
		f'system seen {fmt(avg(s_seen_system, metric))},', 
		f'system unseen {fmt(avg(s_unseen_system, metric))},', 
		f'system all {fmt(avg(s_system, metric))}'
	)

	print(
		f'{metric} std: merged seen {fmt(std(s_seen_merged, metric))},', 
		f'merged unseen {fmt(std(s_unseen_merged, metric))},',
		f'merged all {fmt(std(s_merged, metric))} |', 
		f'system seen {fmt(std(s_seen_system, metric))},', 
		f'system unseen {fmt(std(s_unseen_system, metric))},', 
		f'system all {fmt(std(s_system, metric))}'
	)

	print(
		f'{metric} m>s: seen {fmt(gaussian_sup(avg(s_seen_merged, metric), std(s_seen_merged, metric), avg(s_seen_system, metric), std(s_seen_system, metric)))},', 
		f'unseen {fmt(gaussian_sup(avg(s_unseen_merged, metric), std(s_unseen_merged, metric), avg(s_unseen_system, metric), std(s_unseen_system, metric)))},',
		f'all {fmt(gaussian_sup(avg(s_merged, metric), std(s_merged, metric), avg(s_system, metric), std(s_system, metric)))}'
	)
	print(
		f'{metric} anova p: seen {fmt(anova_pvalue(s_seen_merged, s_seen_system, metric))},',
		f'unseen {fmt(anova_pvalue(s_unseen_merged, s_unseen_system, metric))},',
		f'all {fmt(anova_pvalue(s_merged, s_system, metric))}'
	)
	print(
		f'{metric} paired t p: seen {fmt(paired_ttest_pvalue(s_seen_merged, s_seen_system, metric))},',
		f'unseen {fmt(paired_ttest_pvalue(s_unseen_merged, s_unseen_system, metric))},',
		f'all {fmt(paired_ttest_pvalue(s_merged, s_system, metric))}'
	)
	print(
		f'{metric} pair corr r: seen {fmt(pair_correlation(s_seen_merged, s_seen_system, metric))},',
		f'unseen {fmt(pair_correlation(s_unseen_merged, s_unseen_system, metric))},',
		f'all {fmt(pair_correlation(s_merged, s_system, metric))}'
	)
	print()

	# sum(s[metric] for s in s_merged)/len(s_merged)


# %% OLD EVAL FOR COMPARISON

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
