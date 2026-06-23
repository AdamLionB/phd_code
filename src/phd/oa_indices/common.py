from .diversity_indices import diversity_eval
from .performance_indices import P_R_F, P_R_F_and_base_numbers
from ..cupt_parser import DataFrame, inline_mwes
from ..utils import SENTENCE_ID, MWE_ID, TOKEN_ID, fmset
import pandas as pd
from typing import Iterable, Optional

def full_eval(
	truth: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
	pred: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
) -> dict[str, Optional[float]]:
	inline_truth, inline_pred, inline_true_pred = true_pred(truth, pred)
	p, r, f, tp, pred_count, truth_count = P_R_F_and_base_numbers(inline_truth, inline_pred, inline_true_pred)

	return {'p': p, 'r' : r, 'f' : f, 'tp': tp, 'pred': pred_count, 'truth': truth_count} | diversity_eval(inline_true_pred)
# def diversity_eval(true_pred: DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]]):

def true_pred(
	truth: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
	pred: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
) -> tuple[
	DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	DataFrame[tuple, tuple[SENTENCE_ID, fmset, tuple]]
]:
	inline_truth = inline_mwes(truth)
	inline_pred = inline_mwes(pred)
	true_pred = pd.merge(
		inline_truth.reset_index(),
		inline_pred.reset_index(),
		how='inner'
	)
	return inline_truth, inline_pred, true_pred

def seen_unseen_full_eval(
	truth: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
	pred: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
	seen_mwe_types: Iterable[fmset[str]],
) -> tuple[dict[str, Optional[float]], dict[str, Optional[float]]]:
	
	inline_truth_seen, inline_pred_seen,inline_true_pred_seen,\
	inline_truth_unseen, inline_pred_unseen, inline_true_pred_unseen = seen_unseen_true_pred(truth, pred, seen_mwe_types)

	p_seen, r_seen, f_seen, tp_seen, pred_seen, truth_seen = P_R_F_and_base_numbers(inline_truth_seen, inline_pred_seen, inline_true_pred_seen)
	p_unseen, r_unseen, f_unseen, tp_unseen, pred_unseen, truth_unseen = P_R_F_and_base_numbers(inline_truth_unseen, inline_pred_unseen, inline_true_pred_unseen)

	seen_res = {'p': p_seen, 'r' : r_seen, 'f' : f_seen, 'tp': tp_seen, 'pred': pred_seen, 'truth': truth_seen} | diversity_eval(inline_true_pred_seen)
	unseen_res = {'p': p_unseen, 'r' : r_unseen, 'f' : f_unseen, 'tp': tp_unseen, 'pred': pred_unseen, 'truth': truth_unseen} | diversity_eval(inline_true_pred_unseen)

	return seen_res, unseen_res

def seen_unseen_true_pred(
	truth: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
	pred: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
	seen_mwe_types: Iterable[fmset[str]],
):
	
	inline_truth = inline_mwes(truth)
	# filter inline_truth to keep only lines where inline_truth[0] (the mwe type) is in seen_mwe_types
	inline_truth_seen = inline_truth[inline_truth[0].isin(seen_mwe_types)]
	inline_truth_unseen = inline_truth[~inline_truth[0].isin(seen_mwe_types)]

	inline_pred = inline_mwes(pred)
	inline_pred_seen = inline_pred[inline_pred[0].isin(seen_mwe_types)]
	inline_pred_unseen = inline_pred[~inline_pred[0].isin(seen_mwe_types)]

	true_pred_seen = pd.merge(
		inline_truth_seen.reset_index(),
		inline_pred_seen.reset_index(),
		how='inner'
	)

	true_pred_unseen = pd.merge(
		inline_truth_unseen.reset_index(),
		inline_pred_unseen.reset_index(),
		how='inner'
	)

	return inline_truth_seen, inline_pred_seen, true_pred_seen,\
		inline_truth_unseen, inline_pred_unseen, true_pred_unseen

