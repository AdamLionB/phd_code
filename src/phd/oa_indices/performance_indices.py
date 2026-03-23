from cupt_parser import DataFrame
from lambda_css_utils import SENTENCE_ID, MWE_ID, TOKEN_ID, fmset
# import cupt_parser

def P_R_F(
	inline_truth : DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	inline_pred : DataFrame[tuple[SENTENCE_ID], tuple[fmset, tuple]],
	inline_true_pred : DataFrame[tuple, tuple[SENTENCE_ID, fmset, tuple]]
):
	p = len(inline_true_pred)/len(inline_pred)
	r = len(inline_true_pred)/len(inline_truth)
	f = (2 * p * r) / (p + r)
	return p, r, f
	