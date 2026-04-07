#%%
from typing import Iterator,Callable, Any, Union, Optional, NewType
from conllu import TokenList, parse_incr, TokenTree
import pandas as pd
import os

from itertools import chain, takewhile, count
import re

# import lambda_css_utils as utils
from . import utils
from .utils import DataFrame
from .utils import SENTENCE_ID, MWE_ID, TOKEN_ID
# try  :
# 	from code1 import utils
# 	from code1.utils import DataFrame
# except :
# 	import utils
# 	from utils import DataFrame

# SENTENCE_ID = NewType('SENTENCE_ID', int)
# MWE_ID = NewType('MWE_ID', str)
# TOKEN_ID = NewType('TOKEN_ID', str)

# ----------------------------------
# redefining TokenTrees, the dirty way.
class TT(TokenTree):
	'''Custom hashable tokentree class

	Parameters
	----------
	TokenTree : TokenTree
		[description]
	'''
	def __hash__(self):
		try:
			return hash(frozenset(self.token.items()))
		except:
			print(self.token.items())
			return hash(frozenset(self.token.items()))
		# return hash(frozenset(self.token.items())) + \
		# 	hash(FrozenMultiSet(self.children))
	def __eq__(self, other):
		if not isinstance(other, self.__class__):
			return False
		return (self.token == other.token) and  \
			(utils.fmset(self.children) == utils.fmset(other.children)) #and \
			# (self.metadata == other.metadata)
	def __lt__(self, o):
		return hash(self) < hash(o)
	def __iter__(self):
		yield self
		for child in self.children:
			yield from child

def to_custom_TT(tokentree : TokenTree) -> TT:
	'''Converts a conllu.TokenTree into a custom TokenTree (TT).
	TTs have no dict or list in their token, and 
	redefine the __hash__, __eq__, __lt__, and __iter__ functions.

	'''
	return TT({
			k : v 
			for k,v in tokentree.token.items() 
			if not isinstance(v, dict) and not isinstance(v, list) 
		},
		[to_custom_TT(c) for c in tokentree.children],
		tokentree.metadata)

# ----------------------------------

class Cupt_parser:
	def __init__(
		self, 
		src : Union[str, list[TokenList]],
		preprocessing : Optional[list[Callable[[TokenList], TokenList]]] = None,
		postprocessing : Optional[list[Callable[[DataFrame], DataFrame]]] = None
	):
		'''Setup a parser which takes a .conll / .conllu / .cupt or and list of 
		TokenList and outputs a DataFrame where each feature is a column,
		each word a row, and TokenTree of each word is a the ...

		Parameters
		----------
		src : str | list[TokenList]
			either the path to the .conll / .conllu / .cupt file or list of Tokenlist.
		preprocessing : list[Callable[[TokenList], TokenList]] | None, optional
			list of functions (TokenList) -> TokenList to be executed before the
			Dataframe et TokenTree are generated, by default None
		postprocessing : list[Callable[[DataFrame], DataFrame]] | None, optional
			list of functions (DataFrame) -> DataFrame to be executed after the 
			DataFrame and the TokenTree are generated 
			(usefull when working with batches), by default None

		Raises
		------
		TypeError
			if src's type is incorrect
		'''
		if type(src) == str:
			from . import dcupt
			if dcupt.is_dcupt(src):
				file = dcupt.resolve_as_stream(src)
			else:
				file = open(src, 'r', encoding="utf-8")
			self.parser_TL = parse_incr(file)
		elif type(src) == list:
			self.parser_TL = src
		else :
			raise TypeError
			
		
		self._has_ended = False
		self._counter = 0
		self._preprocessing = preprocessing or [] # same as: preprocessing if preprocessing != None else []
		self._postprocessing = postprocessing or [] # it's magic

	def read_next_n(self, n : int = 0) -> Iterator[TokenList]:
		'''
		Iterates over the next n Tokenlists and apply the preprocessing
		'''
		for i, tl in enumerate(self.parser_TL):
			for f in self._preprocessing:
				tl = f(tl)
			yield tl
			if i >= n-1 and n != 0:
				return
		self._has_ended = True
		
	def get_df_per_batch(self, batch_size : int) -> Iterator[DataFrame]:
		def tt_to_tts(tree : TokenTree) -> list[TokenTree]:
			return [tree] + list(chain.from_iterable(tt_to_tts(child) for child in tree.children))

		def tl_to_ordered_tts(sentence : TokenList):
			return sorted(tt_to_tts(sentence.to_tree()), key = lambda x: x.token['id'])

		while not self._has_ended:
			df = DataFrame.from_dict(
				{
					(self._counter + m, token['id']): { 
						**token, 
						'TT' : next(tt) if type(token['id']) == int else None
					}
					for m, (tl, tt) in enumerate(map(
						lambda x : (
							x, 
							filter(lambda tt: tt.token['id'] != 0, tl_to_ordered_tts(x))
						),
						self.read_next_n(batch_size)
					))
					for token in tl
				},
				orient='index'
			)
			df.index = df.index.rename(['sentence_id', 'token_id'])
			for f in self._postprocessing:
				df = f(df)
			yield df
			self._counter += batch_size
	def get_df(self) -> DataFrame:
		return next(self.get_df_per_batch(0))

	def get_df_no_tt_per_batch(self, batch_size):
		# lines = [
		# 	((self._counter + m, token['id']), {**token})
		# 	for m, tl in enumerate(self.read_next_n(batch_size))
		# 	for token in tl
		# ]
		# index, rows = zip(*lines) if lines else ([], [])
		# df = DataFrame(list(rows), index=index)
		df =  DataFrame.from_dict(
			{
				(self._counter + m, token['id']): {**token}
				for m, tl in enumerate(self.read_next_n(batch_size))
				for token in tl
			},
				orient='index'
		)
		df.index = df.index.rename(['sentence_id', 'token_id'])
		for f in self._postprocessing:
			df = f(df)
		yield df
		self._counter += batch_size
	def get_df_no_tt(self) -> DataFrame:
		return next(self.get_df_no_tt_per_batch(0))
	

def atomize(tl : TokenList) -> TokenList:
	'''
	atomize the column 'feats'
	'''
	for token in tl:
		token['feats'] = token['feats'] or {} # magic
		for k, v in token['feats'].items():
			token[k] = v
		del token['feats']
	return tl

def remove_compound(df : DataFrame) -> DataFrame:
	'''
	removes all rows of the df where 'id' is not an integer
	'''
	return df.loc[df['id'].apply(type) == int]
		

def locmap(df: DataFrame, column: str, func: Callable[[Any], bool]):
	'''function Monkey patched to pd.DataFrame(check on wiki)
	Given a dataframe, a column of said dataframe and a function
	returns the dataframe containing only those rows for which the function
	is True.
	
	Is kinda to `.loc` what `.applymap` is to `.apply`.
	`.loc` only takes function which can be vectorized. This method "fixes" that.
	'''
	return df.loc[lambda x: x[column].apply(func)]
DataFrame.locmap = locmap
pd.DataFrame.locmap = locmap



def setup_data(
	file_path : str = 'dev.cupt',
	preproc_f : Optional[list] = None
) -> tuple[DataFrame[
	tuple[SENTENCE_ID], TT],
	DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple
]]:
	def custom_TTs(df : DataFrame) -> DataFrame:
		df['TT'] = df['TT'].apply(to_custom_TT)
		return df

	preprocessing = [atomize] + (preproc_f if preproc_f else [])
	parser = Cupt_parser(
		file_path,
		preprocessing,
		[
			remove_compound, 
			custom_TTs
		]
	)
	df = parser.get_df()
	TTs = df.loc[df['head'] == 0][['TT']] \
		.droplevel(1) \
		.rename({'TT' : 'sentence'}, axis=1)
	df = df.drop('misc', axis=1)
	df = df.drop('deps', axis=1)
	return TTs, df

def setup_data_noTT(
	file_path : str = 'dev.cupt',
	preproc_f : Optional[list] = None,
	postproc_f : Optional[list] = (remove_compound,),
) -> DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple]:

	preprocessing = [atomize] + (preproc_f if preproc_f else [])
	parser = Cupt_parser(
		file_path,
		preprocessing,
		postproc_f

	)
	df = parser.get_df_no_tt()

	df = df.drop('misc', axis=1)
	df = df.drop('deps', axis=1)
	return df



def remove_NE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(rf'^{i}(;|:|$)', y) == None]) or '*'
	def h(x):
		
		return [
			i
			for i, e, f in takewhile(
				lambda x: len(x[1]) != 0,
				map(
					lambda i : (
						i,
						x.filter(**{
							'parseme:mwe': lambda x: re.search(rf'(;|^){i}(;|:|$)', x) != None
						}),
						x.filter(**{
							'parseme:mwe':
							lambda x: re.search(rf'(?:;|^){i}:\w+?\|(NE).+?\|.+', x) != None
						})
					),
					count(1) 
				)
			)
			if len(f) != 0
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res

def remove_NE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(rf'^{i}(;|:|$)', y) == None]) or '*'
	def h(tl):
		
		return [
			i
			for i in {
				y 
				for x in tl
				for y in re.findall(r'(\d+)', x['parseme:mwe'])
			}
			if tl.filter(**{
				'parseme:mwe':
				lambda x: re.search(rf'(?:;|^){i}:\w+?\|(NE).+?\|.+', x) != None
			})
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res

def remove_VMWE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(rf'^{i}(;|:|$)', y) == None]) or '*'
	def h(tl):
		
		return [
			i
			for i in {
				y 
				for x in tl
				for y in re.findall(r'(\d+)', x['parseme:mwe'])
			}
			if tl.filter(**{
				'parseme:mwe':
				lambda x: re.search(rf'(?:;|^){i}:\w+?\|(MWE-).+?\|.+', x) != None
			})
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res

def remove_nVMWE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(rf'^{i}(;|:|$)', y) == None]) or '*'
	def h(tl):
		
		return [
			i
			for i in {
				y 
				for x in tl
				for y in re.findall(r'(\d+)', x['parseme:mwe'])
			}
			if tl.filter(**{
				'parseme:mwe':
				lambda x: re.search(rf'(?:;|^){i}:\w+?\|(MWE(?!-)).*?\|.+', x) != None
			})
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res


regex_map_cache = {i : re.compile(rf'(;|^){i}(;|:|$)') for i in range(10)}
def regex_map(i):
	if not i in regex_map_cache:
		regex_map_cache[i] = re.compile(rf'(;|^){i}(;|:|$)')
	return regex_map_cache[i]

regex1 = re.compile(r'(\d+)')



def get_mwes(df : DataFrame) -> DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple]:
	'''Returns a dataframe with (sentence_id, token_id, mwe_id) for index,
	with only token from MWEs. If a token is part of multiple MWE it is duplicated.

	'''

	items = [
		locmap(
			sentence,
			'parseme:mwe',
			lambda x: re.search(regex_map(i), x) != None
		).assign(mwe_id=i) # gets the token associated to the number i
		for _, sentence in df.groupby(level=0) # for each sentence
		for i in {
			y
			for x in sentence['parseme:mwe'].apply(
				lambda x: re.findall(regex1, x)
			)
			for y in x
		} # for each number associated to a MWE component in the sentence
	]
	if not items:
		return pd.DataFrame()
	return pd.concat(items).set_index('mwe_id', append=True)

	

# def inline_mwes(
# 	mwes : DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple]
# ) -> DataFrame[tuple[SENTENCE_ID], tuple[utils.fmset, tuple]]:
# 	tmp = mwes.groupby(level=[0, 2]).apply(
# 		lambda x : (utils.fmset(list(x['lemma'])), tuple(x['id']))
# 	).apply(lambda x: pd.Series(x))
# 	return tmp.reset_index(
# 		'mwe_id',
# 		drop=True
# 	).reset_index().drop_duplicates().set_index('sentence_id')
def inline_mwes(
	mwes : DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple]
) -> DataFrame[tuple[SENTENCE_ID], tuple[utils.fmset, tuple]]:
	grouped = mwes.groupby(level=[0, 2]).apply(
		lambda x : (utils.fmset(list(x['lemma'])), tuple(x['id']))
	)
	tmp = pd.DataFrame(
		[[v[0], v[1]] for v in grouped],
		index=grouped.index,
		columns=[0, 1]
	)
	return tmp.reset_index(
		'mwe_id',
		drop=True
	).reset_index().drop_duplicates().set_index('sentence_id')


def write_matches_as_dcupt(
	matches: DataFrame,
	base_blind_path: str,
	output_path: str,
):
	"""Write lexicon match results as a .dcupt file.

	Converts a match DataFrame (with sentence_id, token_id, mwe_id multi-index)
	into a .dcupt file that overrides PARSEME:MWE in the base blind cupt.

	Parameters
	----------
	matches : DataFrame
		Match results with (sentence_id, token_id, mwe_id) multi-index.
	base_blind_path : str
		Path to the base blind .cupt file.
	output_path : str
		Where to write the .dcupt file.
	"""
	from . import dcupt

	# Read sentence structure from base file: list of token ID strings per sentence
	sentences_structure = []
	current = []
	with open(base_blind_path, 'r', encoding='utf-8') as f:
		for line in f:
			stripped = line.strip()
			if not stripped:
				if current:
					sentences_structure.append(current)
					current = []
			elif not stripped.startswith('#'):
				current.append(stripped.split('\t')[0])
	if current:
		sentences_structure.append(current)

	overrides = {}
	if matches.empty:
		for i in range(len(sentences_structure)):
			overrides[(0, i)] = []
	else:
		matched_sents = set(matches.index.get_level_values('sentence_id').unique())
		for sent_idx, token_id_strs in enumerate(sentences_structure):
			if sent_idx not in matched_sents:
				overrides[(0, sent_idx)] = []
				continue

			sent_matches = matches.loc[[sent_idx]]
			sent_matches = sent_matches.droplevel('sentence_id')
			token_mwe_map = {}
			for mwe_num, (mwe_id, group) in enumerate(
				sent_matches.groupby(level='mwe_id'), 1
			):
				tids = sorted(group.index.get_level_values('token_id'))
				for i, tid in enumerate(tids):
					if tid not in token_mwe_map:
						token_mwe_map[tid] = []
					token_mwe_map[tid].append((mwe_num, i == 0))

			annotations = []
			for tid_str in token_id_strs:
				try:
					tid = int(tid_str)
				except (ValueError, TypeError):
					annotations.append('*')
					continue
				if tid in token_mwe_map:
					parts = []
					for mwe_num, is_first in token_mwe_map[tid]:
						parts.append(f'{mwe_num}:MWE' if is_first else str(mwe_num))
					annotations.append(';'.join(parts))
				else:
					annotations.append('*')
			overrides[(0, sent_idx)] = annotations

	base_ref = os.path.relpath(
		os.path.abspath(base_blind_path),
		os.path.dirname(os.path.abspath(output_path))
	)

	dcupt.create(
		output_path=output_path,
		base_ref=base_ref,
		columns=['PARSEME:MWE'],
		default_value='*',
		overrides=overrides,
	)
