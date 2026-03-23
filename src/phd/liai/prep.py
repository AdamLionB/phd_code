#%%
import torch
import pandas as pd

import numpy as np
from .. import cupt_parser
import pandas as pd



def df_to_tensor(df, sentence_len = 115, pad_idx = 0):

	
	df = df.loc[(df.groupby(level=0).apply(len) < sentence_len).loc[lambda x: x].index]

	#FIXME manage larger than sentence_len sentences
	if type(df) == pd.DataFrame:
		filler = [[0]*len(df.columns)]
	else :
		filler = [pad_idx]
	a= df.groupby(level=0).apply(lambda x: x.to_numpy())
	b= df.groupby(level=0).apply(lambda x: filler) * (
		sentence_len - df.groupby(level=0).apply(len)
	)
	b = b.apply(lambda x: np.array(x, dtype=np.int64))
	c = a.combine(b, lambda x, y : np.concatenate([x, y]) if len(y) != 0 else x)
	# x = c[1:2]
	c = np.array([row for row in c], dtype=int)
	# return c
	return torch.from_numpy(
		c
	)


def file2ts(file, voc, max_len = 0, split= 1):
	DATA = cupt_parser.setup_data_noTT(
		file
	)
	if max_len > 0:
		pass #TODO
	else:
		max_len = DATA['id'].groupby(level=0).count().max()
	mwes = cupt_parser.inline_mwes(cupt_parser.get_mwes(DATA))
	
	mask = torch.nn.utils.rnn.pad_sequence(
		DATA.groupby(level=0).apply(lambda x: torch.tensor(x['id'].tolist()) * 0 +1).tolist()
		, batch_first=True
		,padding_value=0
	)
	indices = list(set(DATA.reset_index()['sentence_id']))
	indices_map = {x : n for n, x in enumerate(indices)}

	labels = pd.concat([
		mwes[1].apply(
				lambda x: torch.scatter(
					torch.zeros(max_len),
					0,
					torch.tensor(tuple(x)) - 1, torch.ones(max_len)
				).int().tolist()
			),
		DATA.groupby(level=0).apply(lambda x: torch.zeros(max_len).int().tolist()).rename(2)
	])
	labels = labels.rename(index=indices_map)

	nb_sentences = len(indices_map)
	rng = np.random.default_rng()
	tmp = rng.permutation(np.array(list(indices_map.values())))
	# if split != 0 and split != 1:
	train_ids, test_ids = tmp[:int(nb_sentences * split)], tmp[int(nb_sentences * split):]
	# else:
	# 	train_ids, test_ids = tmp, None
	dev_ids = tmp[:3]

	df = DATA.drop('parseme:mwe', axis=1)[['id', 'lemma', 'head']]#, 'upos', 'head', 'deprel']]
	# df['id'] = pd.to_numeric(DATA['id'], errors = 'coerce')
	df['lemma'] = df['lemma'].apply(lambda x: voc.get(x))
	df['head'] = df['head'].apply(int)
	
	# keymap = get_keymap(df)

	data = df_to_tensor(df, max_len+1)

	sentences = DATA['form'].groupby(level=0).apply(lambda x: list(x))
	sentences = sentences.rename(index=indices_map)
	# data = prep_df_for_tensor(df)

	X_train = data[train_ids]
	Y_train = labels.loc[train_ids]

	if len(test_ids) != 0:
		X_test = data[test_ids]
		Y_test = labels.loc[test_ids]
	else:
		X_test = None
		Y_test = None

	X_dev = data[dev_ids]
	Y_dev = labels.loc[dev_ids]
	return sentences, train_ids, test_ids, data, X_train, Y_train, X_test, Y_test, X_dev, Y_dev, mask

	# x = torch.nn.utils.rnn.pad_sequence(
	# 	DATA.groupby(level=0).apply(lambda x: torch.tensor(x['id'].tolist()) * 0).tolist() + [torch.zeros(115)]
	# 	, batch_first=True
	# 	,padding_value=-1
	# )
	# a = pd.DataFrame(mwes.apply(
	# lambda y: x[y.name].int().scatter(-1, torch.tensor(y[1]).long(), torch.ones(len(y[1])).int())
	# ,axis=1
	# ))
	# b = pd.DataFrame([(y.int(),) for y in x])
	# labels = pd.concat([a, b])


class Voc:
	def __init__(self):
		self.dic = {}
	def get(self, k):
		if not (k in self.dic):
			self.dic[k] = len(self.dic)
		return self.dic[k]