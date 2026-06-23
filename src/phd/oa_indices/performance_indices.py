from ..cupt_parser import DataFrame
from ..utils import SENTENCE_ID, MWE_ID, TOKEN_ID, fmset
import math

# import cupt_parser

def P_R_F(
	inline_truth : DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	inline_pred : DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	inline_true_pred : DataFrame[tuple, tuple[SENTENCE_ID, fmset, tuple]]
):
	p = len(inline_true_pred)/len(inline_pred) if len(inline_pred) > 0 else math.nan
	r = len(inline_true_pred)/len(inline_truth) if len(inline_truth) > 0 else math.nan
	f = (2 * p * r) / (p + r) if p + r > 0 else math.nan
	return p, r, f

def P_R_F_and_base_numbers(
	inline_truth : DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	inline_pred : DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	inline_true_pred : DataFrame[tuple, tuple[SENTENCE_ID, fmset, tuple]]
):
	p = len(inline_true_pred)/len(inline_pred) if len(inline_pred) > 0 else math.nan
	r = len(inline_true_pred)/len(inline_truth) if len(inline_truth) > 0 else math.nan
	f = (2 * p * r) / (p + r) if p + r > 0 else math.nan
	return p, r, f, len(inline_true_pred), len(inline_pred), len(inline_truth)