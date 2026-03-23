from .diversity_indices import diversity_eval
from .performance_indices import P_R_F
from cupt_parser import DataFrame, inline_mwes
from lambda_css_utils import SENTENCE_ID, MWE_ID, TOKEN_ID, fmset
import pandas as pd

def full_eval(
	truth: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
	pred: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple],
):
	inline_truth, inline_pred, inline_true_pred = true_pred(truth, pred)
	p, r, f = P_R_F(inline_truth, inline_pred, inline_true_pred)

	return {'p': p, 'r' : r, 'f' : f} | diversity_eval(inline_true_pred)
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