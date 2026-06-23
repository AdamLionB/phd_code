import pandas as pd
from scipy.stats import normaltest, norm, monte_carlo_test, mannwhitneyu, ttest_ind, wilcoxon
from typing import Optional, Literal
class FlagMeta(type):
	def __repr__(cls):
		return cls._symbol

class Flag(metaclass=FlagMeta):
	_symbol = "?"
	def __str__(self): return self._symbol
	def __repr__(self): return self._symbol
	# def __and__(self, other):
	# 	a_pos = self is POSITIVE
	# 	b_pos = other is POSITIVE
	# 	a_neg = self is NEGATIVE
	# 	b_neg = other is NEGATIVE
	# 	if a_pos and b_pos:
	# 		return POSITIVE
	# 	elif a_neg and b_neg:
	# 		return NEGATIVE
	# 	elif (a_pos and b_neg) or (a_neg and b_pos):
	# 		return INCONCLUSIVE
	# 	else:
		


class REJECTED(Flag): _symbol = "✗"

class NOT_REJECTED(Flag): _symbol = "~"

class INCONCLUSIVE(Flag): _symbol = "?"

class POSITIVE(Flag): _symbol = "✓"

class NOT_POSITIVE(Flag): _symbol = "~"

class NEGATIVE(Flag): _symbol = "✗"

class NOT_NEGATIVE(Flag): _symbol = "~"

ALPHA = 0.05

def reject(p_value) -> Flag:
	if pd.isna(p_value):
		return INCONCLUSIVE
	elif p_value < ALPHA:
		return REJECTED
	elif p_value >= ALPHA:
		return NOT_REJECTED
	else:
		return INCONCLUSIVE



def normality_test(s: pd.Series) -> Optional[float]:
	def statistic(x, axis):
		# Get only the `normaltest` statistic; ignore approximate p-value
		return normaltest(x, axis=axis, nan_policy='omit').statistic

	if s.std() == 0 or pd.isna(s.std()):
		return None

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

def equality_test(
	a: pd.Series,
	b: pd.Series,
	alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
	a_is_normal: Optional[Flag] = None,
	b_is_normal: Optional[Flag] = None
) -> Optional[float]:
	'''
	Perform a t-test or Mann-Whitney U test to compare two samples, depending on their normality.
	'''
	if a_is_normal is None:
		a_is_normal = reject(normality_test(a))
	if b_is_normal is None:
		b_is_normal = reject(normality_test(b))
	try:
		if a_is_normal != NOT_REJECTED or b_is_normal != NOT_REJECTED:
			# If either sample is not normal or results are inconclusive, use Mann-Whitney U test
			_, p = mannwhitneyu(a, b, alternative=alternative, nan_policy='omit')
			return p

		_, p = ttest_ind(a, b, alternative=alternative, nan_policy='omit', equal_var=False)
		return p
	except Exception as e:
		return None

def paired_equality_test(
	a: pd.Series,
	b: pd.Series,
	alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
	a_is_normal: Optional[Flag] = None,
	b_is_normal: Optional[Flag] = None
) -> Optional[float]:
	'''
	Perform a t-test or Mann-Whitney U test to compare two samples, depending on their normality.
	'''
	# if a_is_normal is None:
	# 	a_is_normal = reject(normality_test(a))
	# if b_is_normal is None:
	# 	b_is_normal = reject(normality_test(b))
	# try:
	# 	if a_is_normal != NOT_REJECTED or b_is_normal != NOT_REJECTED:
	# 		# If either sample is not normal or results are inconclusive, use Mann-Whitney U test
	# 		_, p = mannwhitneyu(a, b, alternative=alternative, nan_policy='omit')
	# 		return p
	try:
		_, p = wilcoxon(a, b, alternative=alternative, nan_policy='omit', zero_method='zsplit')
		return p
	except Exception as e:
		return None

