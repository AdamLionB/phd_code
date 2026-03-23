from dataclasses import dataclass
from typing import Iterable, Any, Generic, TypeVar

from .lambdacss import LambdaCSS, LambdaCSS_spec, lCSSs_from_data, lCSS_occ_in_sentences, SENTENCE_ID
from .lexicon import MWE_lexicon
from .utils import DataFrame, df_from_records, sort_column

import pandas as pd

P = TypeVar('P')


@dataclass
class LCSSAdapter(Generic[P]):
	spec: LambdaCSS_spec[P]

	@property
	def properties(self) -> P:
		return self.spec.properties

	def instantiate(self, sentences: DataFrame) -> Iterable[LambdaCSS[P]]:
		lCSSs = lCSSs_from_data(sentences, self.spec)
		return {
			lcss
			for _, lcss in lCSSs['lcss'].items()
		}

	def match(self, lexicon: MWE_lexicon[LambdaCSS[P]], df: DataFrame, *args, **kwargs) -> Any:
		sorted_column_dict = sort_column(df)
		mwes = df_from_records(
			[
				(sentence_id, lcss, tt)
				for lcss in lexicon
				for sentence_id, tt in lCSS_occ_in_sentences(
					sorted_column_dict,
					lcss
				) 
			],
			columns=['sentence_id', 'lcss', 'occ'],
			index=['sentence_id', 'lcss']
		)
		if mwes.empty:
			return pd.DataFrame()
		return reformat(df, mwes)

	def to_json_dict(self) -> dict:
		return self.spec.to_json_dict()
	
	@classmethod
	def adapter_from_json_dict(cls, data: dict) -> 'LCSSAdapter':
		spec = LambdaCSS_spec.from_json_dict(data)
		return cls(spec=spec)

	def entry_from_json_dict(self, data: dict) -> LambdaCSS[P]:
		return LambdaCSS.from_json_dict(data, self.spec)


def reformat(df: DataFrame, mwes: DataFrame) -> DataFrame:
	return pd.concat([
		df.loc[k].loc[
			sorted([x.token['id'] for x in v['occ'] if x.metadata['dummy'] == False])
		].assign(sentence_id=k, mwe_id=n).reset_index().set_index(
			['sentence_id', 'token_id', 'mwe_id']
		)
		for n, ((k, _), v) in enumerate(mwes.iterrows())
	])