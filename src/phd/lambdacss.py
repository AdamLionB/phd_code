#%%
from __future__ import annotations

# standard  lib
from typing import Callable, Iterator, Optional, TypeVar, Type, Any, Sequence, Generic, \
	NewType, Iterable, Union
from dataclasses import dataclass
from itertools import chain, combinations, product, permutations, takewhile, count
from functools import reduce
from collections import defaultdict
from time import time
from os import listdir
from os.path import join as pjoin
import re
from math import exp, log
import subprocess

# big lib
import numpy as np
import pandas as pd
# from pandas.core.frame import DataFrame
from sklearn.model_selection import KFold, ShuffleSplit

# small lib
from conllu import TokenTree, parse_incr, parse_tree, TokenList


# custom lib

from .cupt_parser import TT
from .utils import fmset, DataFrame, df_from_records, sorted_index_loc
from .utils import SENTENCE_ID, MWE_ID, TOKEN_ID
# custom types
# SENTENCE_ID = NewType('SENTENCE_ID', int)
# MWE_ID = NewType('MWE_ID', str)
# TOKEN_ID = NewType('TOKEN_ID', str)

P = TypeVar('P')  # Type for properties dict


@dataclass(frozen=True)
class LambdaCSS_spec(Generic[P]):
	"""
	Specification for LambdaCSS - holds the properties configuration.
	Frozen to be hashable.
	"""
	properties: P

	def to_json_dict(self) -> dict:
		"""Convert to JSON-serializable dict."""
		return {'properties': dict(self.properties)}

	@classmethod
	def from_json_dict(cls, data: dict) -> LambdaCSS_spec:
		"""Reconstruct from JSON dict."""
		return cls(properties=data['properties'])


class LambdaCSS(Generic[P]):
	'''
	Coarse Syntactic Structure 
	An unordered labelled tree generated from a subsequence of words
	'''
	
	def __init__(
		self,
		prop: dict[str, Any],
		children: Iterable[LambdaCSS[P]],
		spec: LambdaCSS_spec[P],
		dummy: bool = False 
	) -> None:
		'''

		Parameters
		----------
		prop : dict[str, Any]
			Dictionary of all properties of the node
		children : Iterable[LambdaCSS]
			Iterable of children of the node
		spec : LambdaCSS_spec[P]
			The specification for this LambdaCSS
		dummy : bool, optional
			Indicates whether the node is a dummy one, by default False
		'''
		self.prop = prop
		self.spec = spec
		self.dummy = dummy
		if isinstance(children, fmset):
			self.children = children
		else:
			self.children = fmset(children)

	@property
	def properties(self) -> P:
		"""Get properties from spec."""
		return self.spec.properties
	
	def verif_integrity(self) -> LambdaCSS[P]:
		self.children = fmset([e.verif_integrity() for e in self.children])
		return self

	@classmethod
	def from_mwe_mdg(
		cl,
		tt: TT,
		mwe_id: MWE_ID,
		spec: LambdaCSS_spec[P],
		flag: bool = False
	) -> LambdaCSS[P]:
		''' Generates a lambda-CSS from a TT, a mdg TT and a mwe_id'''
		t = tt.token
		if mwe_id in tt.token['parseme:mwe']:
			return cl(
				{
					k: (t.get(k) if v or flag else '')
					for k, v in spec.properties.items()
				},
				[
					cl.from_mwe_mdg(child, mwe_id, spec, True)
					for child in tt.children
				],
				spec
			)
		return cl(
			{
				k : ('' if v or not flag else t['deprel'])
				for k, v in spec.properties.items()
			},
			[
				cl.from_mwe_mdg(child, mwe_id, spec, True)
				for child in tt.children
			],
			spec,
			True
		)
	
	def simplify(self, new_spec: LambdaCSS_spec[P]) -> LambdaCSS[P]:
		''' Generates a simpler lambda-CSS from self, keeping only properties in new_spec'''
		return LambdaCSS(
			{
				x: new_spec.properties[x] 
				for x in self.properties.keys() & new_spec.properties.keys()
			},
			[child.simplify(new_spec) for child in self.children],
			new_spec,
			self.dummy
		)
	
	def pprint(self, n=0):
		''' Pretty print of the lambda-CSS '''
		res = ' '*n + '('
		for k, v in self.prop.items():
			res += str(v) + ' '
		
		for e in self.children:
			res += '\n' + e.pprint(n+1)
		res += ' '*n + ')'
		return res
	
	def deepcopy(self) -> LambdaCSS[P]:
		''' Deep copy of the lambda-CSS '''
		return LambdaCSS(self.prop, [c.deepcopy() for c in self.children], self.spec, self.dummy)
	
	def __iter__(self) -> Iterator[LambdaCSS[P]]:
		''' Iterator over the lambda-CSS nodes '''
		yield self
		for child in self.children:
			yield from child

	def __str__(self):
		# return '(' + str(self.prop) + str(self.children) + ')'
		return self.pprint()
	def __repr__(self):
		# return '(' + str(self.prop) + str(self.children) + ')'
		return self.pprint()
	def tuplize(self):
		''' Converts the lambda-CSS to a tuple representation '''
		# print('~', self.children)
		return tuple(sorted(self.prop.items())), tuple(sorted(self.children))
	def __lt__(self, o):
		return hash(self) < hash(o)
	def __eq__(self, o):
		return self.prop == o.prop and self.children == o.children and self.dummy == o.dummy
	def __hash__(self):
		''' Hash function for the lambda-CSS '''
		# if self.__dict__.get('cache') is None:
		# 	self.cache = hash(frozenset(self.prop.items())) + hash(self.children)
		# return self.cache
		return hash(frozenset(self.prop.items())) #+ hash(self.children)
	def kinda_equal(self : LambdaCSS, tt: TT) -> bool:
		# TODO rename to root_match
		''' Checks if the lambda-CSS root node properties are equal to the TokenTree
		root node properties, does not check children '''
		token = tt.token
		for k, v in self.prop.items():
			try :
				if v != '' and v is not None and v != token[k]:
					return False
			except:
				return False
		return True
	
	def expr(self):
		def intern(node):
			return reduce(
				lambda x, y: x + intern(y),
				node.children,
				[node.prop['lemma']]
			)
		return fmset([x for x in intern(self) if x != ''])


# Standard specs
CSS_SPEC = LambdaCSS_spec({'lemma': True, 'upos': True, 'deprel': False})
CANONICAL_CSS_SPEC = LambdaCSS_spec({'deprel': False})


def mdg_from_TT(node: TT) -> dict[MWE_ID, TT]:
	'''
	Extracts minimal dependancy graphs (in the form of a TokenTree) of annotated
	expressions of a TokenTree (sentence)
	'''
	def intern(node : TT, mwe_set : frozenset) -> dict[MWE_ID, TT]:
		'''
		hides default param used for recursion purposes 
		'''
		mwe_ids = {
			x 
			for x in re.split(':|;', node.token['parseme:mwe'])
			if re.match('\d+', x)
		}
		mwe_set |= mwe_ids 
		if node.children:
			tmp = defaultdict(list)
			f = lambda child : intern(child, mwe_set)
			for mwe_children in map(f, node.children):
				if mwe_children:
					for id, child in mwe_children.items():
						tmp[id] += [child]
			return {
				id : TT(
					token=node.token,
					children= tmp[id],
					metadata={'dummy' : not (id in mwe_ids)}
				) #if id in mwe_set or len(tmp[id]) != 0 else tmp[id][0] 
				for id in mwe_ids | set(tmp.keys())
			}
		else:
			if mwe_ids:
				return {id : TT(
					token=node.token,
					children=[],
					metadata={'dummy' : False}
				) for id in mwe_ids}
			else :
				return {}

	def trim(node : TT, mwe_id : MWE_ID):
		''' Trims the tree to keep only the relevant parts for the mwe_id '''
		mwe_ids = {
			x 
			for x in re.split(':|;', node.token['parseme:mwe'])
			if re.match('\d+', x)
		}
		if len(node.children) == 1 and mwe_id not in mwe_ids:
			return trim(node.children[0], mwe_id)
		else:
			return node

	return {k : trim(v, k) for k,v in intern(node, frozenset()).items()}

def mdgs_from_data(
	sentences_TT : DataFrame[tuple[SENTENCE_ID], tuple[TT] ]
) -> DataFrame[tuple[SENTENCE_ID, MWE_ID], tuple[TT]]:
	'''
	Extracts minimal dependency graphs (in the form of a TokenTree) from 
	a dataframe of sentences

	see `mdgs_from_TT`
	'''
	return df_from_records([
		(sentence_id, mwe_id, mdg)
		for sentence_id, sentence in sentences_TT['sentence'].items()
		for mwe_id, mdg in mdg_from_TT(sentence).items()
		if mdg],
		columns=['sentence_id', 'mwe_id', 'mdg'],
		index=['sentence_id', 'mwe_id']
	)

def lCSSs_from_mdgs(
	mdgs: DataFrame[tuple[SENTENCE_ID, MWE_ID], tuple[TT]],
	spec: LambdaCSS_spec
) -> DataFrame[tuple[SENTENCE_ID, MWE_ID], tuple[LambdaCSS]]:
	'''
	Converts a dataframe of Minimal Dependency Graph to 
	a dataframe of lambdaCSS

	see `LambdaCSS.from_TT`
	'''

	return mdgs[['mdg']].assign(
		lcss=mdgs.reset_index('mwe_id').aggregate(
			lambda x: LambdaCSS.from_mwe_mdg(x.iloc[1], x.iloc[0], spec), axis=1
		).values
	).drop('mdg', axis=1)


def lCSSs_from_data(
	data: DataFrame[tuple[SENTENCE_ID], tuple[TT]],
	spec: LambdaCSS_spec
) -> DataFrame[tuple[SENTENCE_ID, MWE_ID], tuple[LambdaCSS]]:
	'''
	Extracts lambda-CSS from a dataframe of sentences

	see `mdgs_from_data` and `lCSSs_from_mdgs`
	'''
	mdgs = mdgs_from_data(data)
	return lCSSs_from_mdgs(mdgs, spec)


def lCSS_occ_in_sentence(
	sentence : TT,
	lCSS : LambdaCSS
) -> list[TT]:
	'''
	Finds all occurrences of a lambda-CSS in a sentence (TokenTree)
	'''
	res = check_lCSS_match_tt(sentence, lCSS)
	for s_child in sentence.children:
		r = lCSS_occ_in_sentence(s_child, lCSS)
		if r:
			res += r
	return res

def lCSSs_occ_in_sentences(
	lCSSs: DataFrame[tuple[LambdaCSS], tuple[int]],
	data: dict[str, pd.Series]
) -> DataFrame[tuple[SENTENCE_ID, LambdaCSS], tuple[TT]]:
	return df_from_records([
		(sid, x, y)
		for x in lCSSs.index
		for sid, y in lCSS_occ_in_sentences(data, x)],
		columns=['sentence_id', 'lcss', 'occ'],
		index=['sentence_id', 'lcss']
	)

def lCSS_occ_in_sentences(
	sid: dict[str, pd.Series],
	lCSS: LambdaCSS,
	verbose=False
) -> list[tuple[SENTENCE_ID, TT]]:
	'''
	Optimized function to find all occurrences of a lambda-CSS in multiple sentences,
	using precomputed sorted series for each column

	Parameters
	----------
	sid : dict[str, pd.Series]
		Dictionnary of column name to pd.Series, each Series is sorted for efficient searching,
		series are in order of number of unique values (more unique values first)
	lCSS : LambdaCSS
		The lambda-CSS to search for in the sentences
	verbose : bool, optional
		does nothing for now, by default False
	'''
	def intern2(sid, lCSS):
		res = set()
		for c_child in lCSS.children:
			for k in sid.keys():
				if k not in c_child.prop:
					continue
				if not res:
					res = set(sorted_index_loc(
						sid[k],
						c_child.prop[k],
						sid['id'].index
					).droplevel(1))
					break
				else:
					res = res.intersection(set(sorted_index_loc(
						sid[k],
						c_child.prop[k],
						sid['id'].index
					).droplevel(1)))
					break
		return res

	tmp1 = None
	for k in sid.keys():
		if k not in lCSS.prop:
			continue
		tmp1 = set(sorted_index_loc(sid[k], lCSS.prop[k], sid['id'].index))
		break

	if lCSS.children:
		x = intern2(sid, lCSS)
		tmp2 = {(a, b) for a, b in tmp1 if a in x}
	else:
		tmp2 = tmp1
	
	
	# for (k, _), v in sid['TT'].loc[tmp2].iteritems():
	# 	if k == 866:
	# 		tmp = check_css_match_tt(v, css)
	# 		if tmp:
	# 			tmp[0].print_tree()
	# 			print([(x.token['id'], x.metadata) for x in tmp[0]])
	# 			print([(x.prop['form'], x.dummy) for x in css])
	# 			# print(tmp[0].metadata)

	res : list[tuple[SENTENCE_ID, TT]]= [
		(k, x)
		for (k, _), v in sid['TT'].loc[list(tmp2)].items()
		for x in check_lCSS_match_tt(v, lCSS)
	]

	# print([x for x in res if x[0] = 786])

	return res


def check_lCSS_match_tt(tt: TT, lcss: LambdaCSS) -> list[TT]:
	
	## Pre checks for performance
	# check if the lambda-CSS root matches the TokenTree root
	if not lcss.kinda_equal(tt):
		return []
	tt_nb_children = len(tt.children)
	lcss_nb_children = len(lcss.children)
		# check number of children are compatible
	if tt_nb_children < lcss_nb_children:
		return []
	
	# simple case : lcss has no children
	if lcss_nb_children == 0:
		# TODO check if we want to keep tt metadata or lcss dummy info
		return [TT(tt.token, [], {'dummy' : lcss.dummy})]
		# return [TT(tt.token, [], tt.metadata)]
	# simple case : both have one child
	if lcss_nb_children == 1 and tt_nb_children == 1:
		# recursive call on the child
		r = check_lCSS_match_tt(tt.children[0], list(lcss.children)[0])
		if r:
			# the child matched, return the corresponding TT
			return [TT(tt.token, [x], {'dummy' : lcss.dummy}) for x in r]
			# return [TT(tt.token, [x], tt.metadata) for x in r]
		return []
	# general case
	# build mapping from (tt_child, lcss_child) to list of matching TT
	m = {
		(tt_child, lcss_child): check_lCSS_match_tt(tt_child, lcss_child)
		for tt_child in tt.children
		for lcss_child in lcss.children 
	}

	lambda1 = lambda x: [m[z] for z in zip(x, lcss.children)]
	
	# TODO refacto ? idk how this work anymore
	# vscode LLM says :
	# we create all the possible combinations of tt children taken lcss_nb_children at a time
	# for each combination, we map each tt child to the list of matching lcss children
	# then we take the product of these lists to get all possible combinations of matching lcss children
	# finally, we filter out the combinations that do not have the correct number of lcss children
	# simpler python code according to vscode LLM :

	return [
		TT(tt.token, list(y), {'dummy' : lcss.dummy})
		# TT(tt.token, list(y), tt.metadata)
		for x in map(lambda1, permutations(tt.children, lcss_nb_children))
		for y in product(*x)
		if len(y) == lcss_nb_children
	]
