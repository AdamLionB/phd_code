#%%
from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, Iterable, Type, Callable, Any, Optional, Protocol
from collections import defaultdict
from itertools import product, groupby
import pickle
import json
import importlib


import pandas as pd
from pandas import DataFrame


from . import cupt_parser

from . import utils
from .utils import SENTENCE_ID, MWE_ID, TOKEN_ID


T = TypeVar('T')
U = TypeVar('U')
S = TypeVar('S', bound='Seq_rep_spec')
E = TypeVar('E', covariant=True)


# ---------- JSON serialization helpers --------

def _get_disc_handler_name(handler: Callable) -> Optional[str]:
	"""Get the name of a discontinuity handler function."""
	if handler is None:
		return None
	return handler.__name__

def _get_disc_handler_by_name(name: Optional[str]) -> Optional[Callable]:
	"""Get a discontinuity handler function by name."""
	if name is None:
		return None
	# Look up in disc_fs list by name
	for f in disc_fs:
		if f.__name__ == name:
			return f
	raise ValueError(f"Unknown discontinuity handler: {name}")


# ---------- Entry type protocol --------

class EntryTypeProtocol(Protocol[E]):
	"""Protocol that entry types must implement."""
	
	def instantiate(self, df: DataFrame) -> Iterable[E]:
		...
	
	def match(self, lexicon: MWE_lexicon[E], df: DataFrame, sid) -> Any:
		...
	
	def to_json_dict(self) -> dict:
		...
	
	@classmethod
	def adapter_from_json_dict(cls, data: dict) -> EntryTypeProtocol[E]:
		...

	def entry_from_json_dict(self, data: dict) -> E:
		...


@dataclass
class SeqRepAdapter(Generic[T, U]):
	spec: Seq_rep_spec[T, U]

	def instantiate(self, df: DataFrame) -> Iterable[Seq_rep]:
		return {
			tmp
			for _, sentence in df.groupby(level=0)
			if len(mwes := cupt_parser.get_mwes(sentence)) != 0
			for _, mwe in mwes.groupby(level=2)
			if (tmp := self.spec.from_mwe(sentence.droplevel(0), mwe))
		}

	def match(self, lexicon: MWE_lexicon[Seq_rep], df: DataFrame, sid):
		tmp = [
			e
			for pattern in lexicon  # Fixed: was `self`, should be `lexicon`
			for e in pattern.match(df, sid)
		]
		if not tmp:
			return pd.DataFrame()
		tmp = pd.concat(
			[x.assign(mwe_id=n) for n, x in enumerate(tmp)]
		)
		tmp = tmp.reset_index()
		try:
			tmp['sentence_id'] = tmp['index'].str[0]
			tmp['token_id'] = tmp['index'].str[1]
		except:
			tmp = tmp.rename({'level_0': 'sentence_id', 'level_1': 'token_id'}, axis=1)
		tmp = tmp.set_index(['sentence_id', 'token_id', 'mwe_id'])
		try:
			tmp = tmp.drop('index', axis=1)
		except:
			pass
		return tmp
	
	def to_json_dict(self) -> dict:
		return self.spec.to_json_dict()
	
	@classmethod
	def adapter_from_json_dict(cls, data: dict) -> SeqRepAdapter:
		return cls(spec=Seq_rep_spec.from_json_dict(data))

	def entry_from_json_dict(self, data: dict) -> Seq_rep:
		return Seq_rep.from_json_dict(data, self.spec)


class MWE_lexicon(Generic[T]):
	def __init__(
		self,
		content: Iterable[T],
		entry_type: EntryTypeProtocol[T],
	):
		self.content = content if isinstance(content, (list, set)) else list(content)
		self.entry_type = entry_type

	def __iter__(self):
		yield from self.content

	def __str__(self) -> str:
		return str(self.content)

	def __repr__(self) -> str:
		return str(self)

	def __len__(self):
		return len(self.content)

	@property
	def T_cl(self) -> EntryTypeProtocol[T]:
		return self.entry_type

	def match(self, *args, **kwargs):
		return self.entry_type.match(self, *args, **kwargs)

	def pickle(self, file_path) -> None:
		with open(file_path, 'wb') as f:
			pickle.dump((self.content, self.entry_type), f, 0)

	@classmethod
	def unpickle(cls, file_path: str) -> MWE_lexicon[T]:
		with open(file_path, 'rb') as f:
			content, entry_type = pickle.load(f)
		return cls(content, entry_type)

	def to_json(self, file_path: str) -> None:
		adapter_class = type(self.entry_type)
		data = {
			'adapter_module': adapter_class.__module__,
			'adapter_class': adapter_class.__name__,
			'entry_type': self.entry_type.to_json_dict(),
			'content': [entry.to_json_dict() for entry in self.content]
		}
		with open(file_path, 'w', encoding='utf-8') as f:
			json.dump(data, f, ensure_ascii=False, indent=2)

	@classmethod
	def from_json(cls, file_path: str) -> MWE_lexicon:
		with open(file_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		
		module = importlib.import_module(data['adapter_module'])
		adapter_class = getattr(module, data['adapter_class'])
		entry_type = adapter_class.adapter_from_json_dict(data['entry_type'])
		content = [entry_type.entry_from_json_dict(entry_data) for entry_data in data['content']]
		
		return cls(content, entry_type)


class Seq_rep(Generic[S]):
	def __init__(
		self,
		components: list,
		insertions: list,
		spec: S,
	):
		self.components = components
		self.insertions = insertions
		self.spec = spec

	def __eq__(self, o: object) -> bool:
		if type(o) == type(self):
			return (
				self.components == o.components 
				and self.insertions == o.insertions
				and self.spec == o.spec
			)
		return False
	
	def __hash__(self) -> int:
		return (
			hash(tuple([frozenset(x.items()) for x in self.components]))
			+ hash(tuple(self.insertions))
		)

	def __str__(self) -> str:
		return str((self.components, self.insertions))
	
	def __repr__(self) -> str:
		return str(self)

	def match(self, df, sid):
		return self.spec.match_entry(self, df, sid)

	def to_json_dict(self) -> dict:
		return {
			'components': self.components,
			'insertions': list(self.insertions),
		}

	@classmethod
	def from_json_dict(cls, data: dict, spec: Seq_rep_spec) -> Seq_rep:
		return cls(
			components=data['components'],
			insertions=[tuple(i) if isinstance(i, list) else i for i in data['insertions']],
			spec=spec
		)


@dataclass
class Seq_rep_spec(Generic[T, U]):
	properties: T
	discontinuity_handler: Callable[[DataFrame[tuple[TOKEN_ID], tuple]], U] = None

	def handle_discontinuities(self, insertions: DataFrame[tuple[TOKEN_ID], tuple]) -> U:
		if self.discontinuity_handler is None:
			return '*'
		return self.discontinuity_handler(insertions)

	def from_mwe(
		self,
		sentence: DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple],
		mwe: DataFrame,
	) -> Optional[Seq_rep[Seq_rep_spec[T, U]]]:
		components = []
		insertions = []
		last_id = mwe.iloc[0]['id']
		for _, row in mwe.iterrows():
			if row['id'] - last_id >= 1:
				try:
					ins_df = sentence.loc[last_id+1:row['id']-1]
					insertions.append(self.handle_discontinuities(ins_df))
				except InvalidInsertion:
					return None
			components.append({prop: row[prop] for prop in self.properties})
			last_id = row['id']
		return Seq_rep(components, insertions, self)

	def match_entry(self, entry: Seq_rep, df, sid):
		ress = [[]]*len(entry.components)
		for n, component in enumerate(entry.components):
			for k,v in component.items():
				if not ress[n]:
					ress[n] = set(utils.sorted_index_loc(sid[k], v, sid['id'].index))
				else:
					ress[n] = ress[n].intersection(
						set(utils.sorted_index_loc(sid[k], v, sid['id'].index))
					)

		matches = [[
				list(v)
				for k, v in groupby(sorted(a, key= lambda x: x[0]), key= lambda x: x[0])
			] for a in ress
		]
		sentences= [defaultdict(list) for _ in range(len(entry.components))]
		for n, x in enumerate(matches):
			for y in x:
				sentences[n][y[0][0]]+=y
		
		res=[]
		keys = set.intersection(*[set(x.keys()) for x in sentences])
		for k in keys:
			for prod in product(*[x[k] for x in sentences]):
				if [e[1] for e in prod] != sorted([e[1] for e in prod]):
					continue

				prod = [df.loc[e] for e in prod]
				for insertion, (a, b) in zip(entry.insertions, zip(prod[:-1],prod[1:])):
					if b['id'] - a['id'] > 1:
						try:
							tmp = self.handle_discontinuities(df.loc[a.name[0]].loc[a['id']+1:b['id']-1])
							if insertion != tmp:
								break
						except InvalidInsertion:
							break
				else:
					res.append(pd.DataFrame(prod))
		return res

	def to_json_dict(self) -> dict:
		return {
			'properties': list(self.properties) if isinstance(self.properties, tuple) else self.properties,
			'discontinuity_handler': _get_disc_handler_name(self.discontinuity_handler),
		}

	@classmethod
	def from_json_dict(cls, data: dict) -> Seq_rep_spec:
		return cls(
			properties=tuple(data['properties']),
			discontinuity_handler=_get_disc_handler_by_name(data['discontinuity_handler']),
		)


class InvalidInsertion(Exception): ...

# Convert disc_* functions from classmethods to regular functions
def disc0(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'contiguous'
	if len(insertions) > 0:
		raise InvalidInsertion

def disc_lemma1(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 lemma max'''
	if len(insertions) > 1:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

def disc_lemma2(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''2 lemma max'''
	if len(insertions) > 2:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

def disc_lemma3(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''3 lemma max'''
	if len(insertions) > 3:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

def disc_lemma4(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''4 lemma max'''
	if len(insertions) > 4:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

def disc_lemma5(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''5 lemma max'''
	if len(insertions) > 5:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

def disc_lemma0(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''list lemma'''
	return tuple(insertions['lemma'])


def disc_pos1(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 pos max'''
	if len(insertions) > 1:
		raise InvalidInsertion
	return tuple(insertions['upos'])

def disc_pos2(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''2 pos max'''
	if len(insertions) > 2:
		raise InvalidInsertion
	return tuple(insertions['upos'])

def disc_pos3(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''3 pos max'''
	if len(insertions) > 3:
		raise InvalidInsertion
	return tuple(insertions['upos'])

def disc_pos4(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''4 pos max'''
	if len(insertions) > 4:
		raise InvalidInsertion
	return tuple(insertions['upos'])

def disc_pos5(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''5 pos max'''
	if len(insertions) > 5:
		raise InvalidInsertion
	return tuple(insertions['upos'])

def disc_pos0(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''list pos'''
	return tuple(insertions['upos'])


def disc_1(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 1 max'''
	if len(insertions) > 1:
		raise InvalidInsertion
	return '*'

def disc_2(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 2 max'''
	if len(insertions) > 2:
		raise InvalidInsertion
	return '*'

def disc_3(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 3 max'''
	if len(insertions) > 3:
		raise InvalidInsertion
	return '*'

def disc_4(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 4 max'''
	if len(insertions) > 4:
		raise InvalidInsertion
	return '*'

def disc_5(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 5 max'''
	if len(insertions) > 5:
		raise InvalidInsertion
	return '*'

def disc_0(insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* unlimited'''
	return '*'

disc_fs = [disc0, 
			disc_lemma1, disc_lemma2, disc_lemma3, disc_lemma4, disc_lemma5, disc_lemma0,
			disc_pos1, disc_pos2, disc_pos3, disc_pos4, disc_pos5, disc_pos0,
			disc_1, disc_2, disc_3, disc_4, disc_5, disc_0]


def instantiate_lexicon(entry_type: EntryTypeProtocol[T], df: DataFrame) -> MWE_lexicon[T]:
	content = entry_type.instantiate(df)
	return MWE_lexicon(content, entry_type)



