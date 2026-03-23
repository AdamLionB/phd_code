#%%
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from .. import cupt_parser
from .. import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import model_copy
from . import prep

# from . import utils
from collections import Counter

import argparse
import wandb
# %%

EPOCH = 50
lr = 0.0001
wd = 0.0001
# wd = 0
pw = 1
bs= 30
load = None
save = None
train = False
eval  = False
ceval = False
wnb = False
mode = 'normal'
frozen = False
LANG = 'Multi'


if __name__=="__main__":
	try:
		parser = argparse.ArgumentParser()

		parser.add_argument("--train", action='store_true')
		parser.add_argument("--eval", action='store_true')
		parser.add_argument("--ceval", action='store_true')
		parser.add_argument("--wandb", action='store_true')
		parser.add_argument("--frozen", action='store_true')
		parser.add_argument("--epoch")
		parser.add_argument("--bs")
		parser.add_argument("--conf")
		parser.add_argument("--rec")
		parser.add_argument("--lr")
		parser.add_argument("--wd")
		parser.add_argument("--pw")
		parser.add_argument("--nothing")
		parser.add_argument("--load")
		parser.add_argument("--save")
		parser.add_argument("--lang")
		parser.add_argument("--mode")
		args = parser.parse_args()
		if args.train:
			train = args.train
		if args.eval:
			eval = args.eval
		if args.ceval:
			ceval = args.ceval
		if args.wandb:
			wnb = args.wandb
		if args.frozen:
			frozen = args.frozen
		if args.epoch:
			EPOCH = int(args.epoch)
		# if args.conf:
		# 	conf = args.conf
		# if args.rec:
		# 	rec = args.rec
		if args.bs:
			bs = int(args.bs)
		if args.load:
			load = args.load
		if args.save:
			save = args.save
		if args.lang:
			LANG = args.lang
		if args.mode:
			mode = args.mode
		if args.lr:
			lr = float(args.lr)
		if args.wd:
			wd = float(args.wd)
		if args.pw:
			pw = float(args.pw)
	except:
		pass

# wd = 0.0001

config = {
  'lr': lr,
  'wd': wd,
  'epochs': EPOCH,
#   'conf': conf,
#   'rec': rec,
  'bs': bs,
  'loss':'bce',
  'pw': pw,
  'frozen': frozen,
  'mode': mode
}

device = 'cpu'
bce = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pw))


class Merger4(nn.Module):
	def __init__(self, nb_annotateurs, frozen, device) -> None:
		super(Merger4, self).__init__()
		self.device = device
		self.bert = model_copy.Bert(device, shorten=True)
		self.classifier = nn.Sequential(
			nn.Linear(768, 100),
			nn.Tanh(),
			nn.Linear(100, 1)
		)
		# self.classifier = nn.Sequential(
		# 	nn.Linear(768, 100),
		# 	nn.Tanh(),
		# 	nn.Linear(100, 1)
		# )
		# self.x = nn.Sequential(
		# 	nn.TransformerEncoder(nn.TransformerEncoderLayer(nb_annotateurs, 4), 2),
		# 	nn.Tanh(),
		# )

		# self.classifier = nn.Linear(768, 1)
		self.bert.partial_freeze(frozen=frozen)
		# self.l1 = nn.Linear(10)
	def forward(self, sentence, *annotations):
		emb = self.bert(sentence)
		# annotations = [x[:,:emb.shape[1],:] for x in annotations]
		
		# tmp = self.x(torch.concat([*annotations], dim=-1))

		out = self.classifier(
			emb
			# torch.concat([emb, tmp], dim=-1)
			# tmp
		)
		out = out.squeeze().mean(dim=-1)
		return out

class Merger3(nn.Module):
	def __init__(self, nb_annotateurs, frozen, device) -> None:
		super(Merger3, self).__init__()
		self.device = device
		# self.bert = model_copy.Bert(device, shorten=True)
		# self.classifier = nn.Sequential(
		# 	nn.Linear(768 + nb_annotateurs, 100),
		# 	nn.Tanh(),
		# 	nn.Linear(100, 1)
		# )
		self.classifier = nn.Sequential(
			nn.Linear(nb_annotateurs, 4),
			nn.Tanh(),
			nn.Linear(4, 1)
		)
		self.x = nn.Sequential(
			nn.Identity()
			# nn.TransformerEncoder(nn.TransformerEncoderLayer(nb_annotateurs, 4), 2),
			# nn.Tanh(),
		)

		# self.classifier = nn.Linear(768 + nb_annotateurs, 1)
		# self.bert.partial_freeze(frozen=frozen)
		# self.l1 = nn.Linear(10)
	def forward(self, sentence, *annotations):
		# emb = self.bert(sentence)
		# annotations = [x[:,:emb.shape[1],:] for x in annotations]
		annotations = [x[:,:,:] for x in annotations]
		
		tmp = self.x(torch.concat([*annotations], dim=-1))

		sizes = [len(x) for x in sentence]
		mask = torch.zeros(len(sizes), tmp.shape[1], device=self.device)
		for i, num_ones in enumerate(sizes):
			mask[i, :num_ones] = 1


		out = self.classifier(
			# torch.concat([emb, tmp], dim=-1)
			tmp
		)
		out = out.squeeze(-1)
		out = (out* mask).mean(dim=-1)
		return out

class Merger2(nn.Module):
	def __init__(self, nb_annotateurs, frozen, device) -> None:
		super(Merger2, self).__init__()
		self.device = device
		self.bert = model_copy.Bert(device, shorten=True)
		self.classifier = nn.Sequential(
			nn.Linear(768 + nb_annotateurs, 100),
			nn.Tanh(),
			nn.Linear(100, 1)
		)
		# self.classifier = nn.Sequential(
		# 	nn.Linear(nb_annotateurs, 4),
		# 	nn.Tanh(),
		# 	nn.Linear(4, 1)
		# )
		self.x = nn.Sequential(
			nn.TransformerEncoder(nn.TransformerEncoderLayer(nb_annotateurs, 4), 2),
			nn.Tanh(),
		)

		# self.classifier = nn.Linear(768 + nb_annotateurs, 1)
		self.bert.partial_freeze(frozen=frozen)
		# self.l1 = nn.Linear(10)
	def forward(self, sentence, *annotations):
		emb = self.bert(sentence)
		annotations = [x[:,:emb.shape[1],:] for x in annotations]
		
		tmp = self.x(torch.concat([*annotations], dim=-1))

		sizes = [len(x) for x in sentence]
		mask = torch.zeros(len(sizes), emb.shape[1], device=self.device)
		for i, num_ones in enumerate(sizes):
			mask[i, :num_ones] = 1

		out = self.classifier(
			torch.concat([emb, tmp], dim=-1)
			# tmp
		)
		out = out.squeeze(-1)
		out = (out* mask).mean(dim=-1)
		return out

class Merger_emb(nn.Module):
	def __init__(self, voc, nb_annotateurs, frozen, device) -> None:
		super(Merger_emb, self).__init__()
		self.device = device
		# self.bert = model_copy.Bert(device, shorten=True)
		self.voc = voc
		# print(len(voc.dic))
		self.emb = nn.Embedding(len(voc.dic)+100, 100)
		self.classifier = nn.Sequential(
			# nn.TransformerEncoder(nn.TransformerEncoderLayer(768 + nb_annotateurs, 4), 2),
			# nn.Tanh(),
			nn.Linear(100 + nb_annotateurs, 50),
			nn.Tanh(),
			nn.Linear(50, 1)
		)
		# self.classifier = nn.Linear(768 + nb_annotateurs, 1)
		# self.bert.partial_freeze(frozen=frozen)
		# self.l1 = nn.Linear(10)
	def forward(self, sentence, *annotations):
		max_len = max(len(s) for s in sentence.to_list())
		# print([[(n, s[n], voc.get(s[n])) if n < len(s) else voc.get('') for n in range(max_len)] for s in sentence.tolist()])
		sentence = [[voc.get(s[n]) if n < len(s) else voc.get('') for n in range(max_len)] for s in sentence.tolist()]
		# print(sentence)
		emb = self.emb(torch.tensor(sentence, device=self.device))
		annotations = [x[:,:emb.shape[1],:] for x in annotations]
		# print(emb.shape)
		sizes = [len(x) for x in sentence]
		mask = torch.zeros(len(sizes), emb.shape[1], device=self.device)
		for i, num_ones in enumerate(sizes):
			mask[i, :num_ones] = 1
		# print(torch.concat([emb, *annotations], dim=-1).shape)
		out = self.classifier(
			torch.concat([emb, *annotations], dim=-1)
		)
		out = out.squeeze(-1)
		out = (out* mask).mean(dim=-1)
		return out
		

class Merger_mtlb(nn.Module):
	def __init__(self, nb_annotateurs, frozen, device) -> None:
		super(Merger_mtlb, self).__init__()
		self.device = device
		self.bert = model_copy.Bert(device, shorten=True)
		self.classifier = nn.Sequential(
			# nn.TransformerEncoder(nn.TransformerEncoderLayer(768 + nb_annotateurs, 4), 2),
			# nn.Tanh(),
			nn.Linear(768 + 1, 100),
			nn.Tanh(),
			nn.Linear(100, 1)
		)
		# self.classifier = nn.Linear(768 + nb_annotateurs, 1)
		self.bert.partial_freeze(frozen=frozen)
		# self.l1 = nn.Linear(10)
	def forward(self, sentence, *annotations):
		emb = self.bert(sentence)
		annotations = [x[:,:emb.shape[1],:] for x in annotations[:1]]
		

		sizes = [len(x) for x in sentence]
		mask = torch.zeros(len(sizes), emb.shape[1], device=self.device)
		for i, num_ones in enumerate(sizes):
			mask[i, :num_ones] = 1
		out = self.classifier(
			torch.concat([emb, *annotations], dim=-1)
		)
		# out = out.squeeze().mean(dim=-1)
		out = out.squeeze(-1)
		out = (out* mask).mean(dim=-1)
		return out

class Merger_bert2(nn.Module):
	def __init__(self, nb_annotateurs, frozen, device) -> None:
		super(Merger_bert2, self).__init__()
		self.device = device
		self.bert = model_copy.Bert2(device, shorten=True)
		self.classifier = nn.Sequential(
			# nn.TransformerEncoder(nn.TransformerEncoderLayer(768 + nb_annotateurs, 4), 2),
			# nn.Tanh(),
			nn.Linear(768 + nb_annotateurs, 100),
			nn.Tanh(),
			nn.Linear(100, 1)
		)
		# self.classifier = nn.Linear(768 + nb_annotateurs, 1)
		self.bert.partial_freeze(frozen=frozen)
		# self.l1 = nn.Linear(10)
	def forward(self, sentence, *annotations):

		emb = self.bert(sentence)
		sizes = [len(x) for x in sentence]
		mask = torch.zeros(len(sizes), emb.shape[1], device=self.device)
		for i, num_ones in enumerate(sizes):
			mask[i, :num_ones] = 1
		annotations = [x[:,:emb.shape[1],:] for x in annotations]
		out = self.classifier(
			torch.concat([emb, *annotations], dim=-1)
		)

		out = out.squeeze(-1)
		out = (out* mask).mean(dim=-1)
		return out

class Merger(nn.Module):
	def __init__(self, nb_annotateurs, frozen, device) -> None:
		super(Merger, self).__init__()
		self.device = device
		self.bert = model_copy.Bert(device, shorten=True)
		self.classifier = nn.Sequential(
			# nn.TransformerEncoder(nn.TransformerEncoderLayer(768 + nb_annotateurs, 4), 2),
			# nn.Tanh(),
			nn.Linear(768 + nb_annotateurs, 100),
			nn.Tanh(),
			nn.Linear(100, 1)
		)
		# self.classifier = nn.Linear(768 + nb_annotateurs, 1)
		self.bert.partial_freeze(frozen=frozen)
		# self.l1 = nn.Linear(10)
	def forward(self, sentence, *annotations):
		emb = self.bert(sentence)
		annotations = [x[:,:emb.shape[1],:] for x in annotations]
		
		out = self.classifier(
			torch.concat([emb, *annotations], dim=-1)
		)
		out = out.squeeze().mean(dim=-1)
		return out
	
def f(a, b, truth):
	tmp = pd.merge(
		a.loc[a.apply(lambda x: 1 in x)].apply(tuple).to_frame().reset_index(),
		b.loc[b.apply(lambda x: 1 in x)].apply(tuple).to_frame().reset_index(),
		how='outer',
		indicator=True,
		left_on=[0, 'sentence_id'],
		right_on=[0, 'sentence_id'],
		sort=True
	)
	return (
		pd.merge(
			tmp,
			truth[truth.apply(lambda x: 1 in x)].apply(tuple).to_frame().reset_index(),
			how='outer',
			indicator='truth',
			left_on=[0, 'sentence_id'],
			right_on=[0, 'sentence_id'],
			sort=True
		).dropna()['truth'].apply(lambda x: (x == 'both') | (x == 'right_only')),
		tmp.transform(
			lambda x: [
				x['sentence_id'],
				x[0] if x['_merge'] == 'left_only' or x['_merge'] == 'both' else None,
				x[0] if x['_merge'] == 'right_only' or x['_merge'] == 'both' else None
			], axis=1
		).fillna(-1).apply(lambda col: col.map(lambda x: x if x != -1 else tuple([0]*len(a.iloc[0]))))
	)

max_f = 0
from math import log

def normalize(p):
	if np.isclose(sum(p), 1):
		return p
	return np.array(p) / sum(p)

def Na(a, p):
	b = 2
	p = normalize(p)
	if a == 1:
		return 2 ** (- sum(x * log(x, b) if x != 0 else 0 for x in p))
	try :
		return sum(x ** a for x in p) ** (1 / (1 - a))
	except ZeroDivisionError:
		return 0

def E_mcl(f):
	N = sum(f)
	return (N - ((sum(x ** 2 for x in f)) ** 0.5)) / (N - (N / (Na(0, f) ** 0.5)))

def E_1mD(p):
	p = normalize(p)
	try :
		return (1 - (Na(2, p) ** -1)) / (1 - (Na(0, p) ** -1))
	except ZeroDivisionError:
		return 0

from scipy.optimize import curve_fit
from scipy.stats import zipfian
def expS(p):
	p = p / sum(p)
	return np.exp(-curve_fit(
			lambda x, s: zipfian.pmf(x, s, len(p)),
			list(range(1, len(p)+1)),
			p.sort_values(ascending=False)
		)[0][0])

def eval_model(model, sentences_test, data_test=None, truth_test=None, Y_system_test=None, Y_lex_test=None, Y_truth_test=None, df_test=None, device='cpu'):
	model.eval()
	c = Counter()
	with torch.inference_mode():
		flag = False
		if data_test is None:
			truth_test, data_test = f(Y_system_test, Y_lex_test, Y_truth_test)
			flag = True
		score= 0
		tmp = np.arange(len(data_test))
		# print(data_test.head(20))
		# print(truth_test)
		# print(sentences_test)
		tp =0
		x = 0
		y = 0
		z = 0
		rloss = 0
		# batch_size = 100
		for i in tqdm(range(len(tmp)//bs+1)):
			batch_ids = tmp[i*bs:i*bs+bs]
			# print(batch_ids)
			if len(batch_ids) == 0:
				# print('*')
				continue
			# print(len(sentences_test), batch_ids)

			batch_test = data_test.iloc[batch_ids]
			# print(batch_test)
			batch_sentences = sentences_test.loc[batch_test['sentence_id']]
			batch_truth_test = torch.tensor(truth_test.iloc[batch_ids].values.astype(int)).to(device)
			# print(len(tmp[i*bs:i*bs+bs]), len(batch_truth_test), len(batch_sentences))
			# print(batch_sentences)
			# print(batch_truth_test)

			r_test = merger_preprocessing(batch_sentences, batch_test, model, device)

			loss = bce(r_test, batch_truth_test.float())	
			rloss += loss.item()
			if df_test is not None:
				c.update([utils.fmset(
						df_test.loc[[
								(v['sentence_id'], n) 
								for n, (x,y) in enumerate(zip(v[0], v['_merge']), 1) 
								if 1 - ((1 - x) * (1 - y)) == 1]]['lemma'].tolist()) 
							for (_, v), r, t in zip(batch_test.iterrows(), r_test, batch_truth_test) 
							if r > 0 and t])
			# print(len(c))
			# print(c)
			tp += torch.sum((r_test > 0).int() & batch_truth_test)
			x += torch.sum((r_test > 0).int())
			z += torch.sum(batch_truth_test)
			# y += sum(batch_truth_test.int())

			# annotated = data_test.iloc[batch_ids].loc[[x for x, y in zip(data_test.iloc[batch_ids].index, r_test.cpu() > 0) if y]]
			# print(tp.item(), x.item(), z.item())
		# print()
		# y = len(truth_test.loc[lambda x: x])
		# print(c)
		if not flag:
			y = len(Y_truth[lambda x: x.apply(lambda x: 1 in x)]) // 4
		else:
			y = len(Y_truth_test[lambda x: x.apply(lambda x: 1 in x)])
		p = (tp / x).item() if x.item != 0 else 0
		r = (tp / y).item()
		
		if wnb:
			wandb.log({
				'loss_t': rloss,
				'p_t': p,
				'r_t': r,
				'f_t': (2 * p * r) / (p + r) if r+p != 0 else 0,
				'tp_t': tp,
				'predicted_t': x,
				'to_predct_t': y,
				'max_r_t': z / y,
				'max_f_t': (2 * (z / y)) / ((z / y) + 1)
			})
		# if save and ((2 * p * r) / (p + r) if r+p != 0 else 0) > 0.6742:
		# 	torch.save(model.state_dict(), str(epoch)+'_'+save)
		if save and ((2 * p * r) / (p + r) if r+p != 0 else 0) > max_f:
			max_f = ((2 * p * r) / (p + r) if r+p != 0 else 0)
			torch.save(model.state_dict(), str(epoch)+'_'+save)


	print(p * 100, r * 100, ((2 * p * r) / (p + r)) * 100 if r+p != 0 else 0, E_1mD(pd.Series(c)), len(c), expS(pd.Series(c)))#, x.item(), y, tp.item(), z.item())
# import .prep

def merger_preprocessing(sentences, data, model, device):
	a, b = torch.split(
		torch.tensor(
			data[[0, '_merge']].apply(lambda col: col.map(list)).values.tolist(), device=device
		).transpose(-1, -2), split_size_or_sections=1, dim=-1
	)
	# print(a.shape, b.shape, len(sentences))
	return  model(
		sentences,
		# sentences_test[batch_test['sentence_id']],
		a,
		b,
		# (a.bool() & ~b.bool()).float(),
		# (~a.bool() & b.bool()).float(),
		(a & b).float(),
		(a | b).float()
	)



voc = prep.Voc()

# sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts('Multi/dev1.system.cupt', voc, 0, 1)
# _,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts('Multi/dev1.blind.cupt.lex', voc, 0, 1)
# _,_,_,_,_,Y_truth,_,_,_,_,_ = prep.file2ts('Multi/dev1.cupt', voc, 0, 1)

# test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts('Multi/dev2.system.cupt', voc, 0, 1)
# _,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts('Multi/dev2.blind.cupt.lex', voc, 0, 1)
# _,_,_,_,_,Y_truth_test,_,_,_,_,_ = prep.file2ts('Multi/dev2.cupt', voc, 0, 1)

# sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts('Multi/dev.system.cupt', voc, 0, 1)
# _,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts('Multi/dev.blind.cupt.lex', voc, 0, 1)
# _,_,_,_,_,Y_truth,_,_,_,_,_ = prep.file2ts('Multi/dev.cupt', voc, 0, 1)



# sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts('dev.system.cupt', voc, 0, 1)
# _,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts('dev.blind.cupt.wesh', voc, 0, 1)
# _,_,_,_,_,Y_truth,_,_,_,_,_ = prep.file2ts('dev.cupt', voc, 0, 1)

# # test mtlb
# sentences_test,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts('test.system.cupt', voc, 0, 1)
# # test lexicon
# _,_,_,_,_,Y_lex_test,_,_,_,_,_ = prep.file2ts('test.blind.cupt.wesh', voc, 0, 1)
# # test truth
# _,_,_,_,_,Y_truth_test,_,_,_,_,_ = prep.file2ts('test.cupt', voc, 0, 1)
	

# #####
# test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts(LANG+'/dev2.system.cupt', voc, 0, 1)
# # _,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/dev2.blind.cupt.lex', voc, 0, 1)
# _,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/dev2.system.cupt', voc, 0, 1)
# _,_,_,_,_,Y_truth_test,_,_,_,_,_ = prep.file2ts(LANG+'/dev2.cupt', voc, 0, 1)


if __name__=="__main__":

	DEVorTEST = 'test'

	if mode == 'normal':
		test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts(LANG+'/'+DEVorTEST+'.system.cupt', voc, 0, 1)
		_,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/'+DEVorTEST+'.blind.cupt.lex', voc, 0, 1)

	if mode == 'lex':
		test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts(LANG+'/'+DEVorTEST+'.blind.cupt.lex', voc, 0, 1)
		_,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/'+DEVorTEST+'.blind.cupt.lex', voc, 0, 1)

	if mode == 'mtlb':
		test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts(LANG+'/'+DEVorTEST+'.system.cupt', voc, 0, 1)
		_,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/'+DEVorTEST+'.system.cupt', voc, 0, 1)

	_,_,_,_,_,Y_truth_test,_,_,_,_,_ = prep.file2ts(LANG+'/'+DEVorTEST+'.cupt', voc, 0, 1)
	df_test = cupt_parser.setup_data_noTT(LANG+'/'+DEVorTEST+'.cupt')

# test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts(LANG+'/test.system.cupt', voc, 0, 1)
# # test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts(LANG+'/test.blind.cupt.lex', voc, 0, 1) # ---
# _,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/test.blind.cupt.lex', voc, 0, 1)
# # _,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/test.system.cupt', voc, 0, 1) # ---
# _,_,_,_,_,Y_truth_test,_,_,_,_,_ = prep.file2ts(LANG+'/test.cupt', voc, 0, 1)

# test_sentences,_,_,_,_,Y_system_test,_,_,_,_,_ = prep.file2ts(LANG+'/test.system.cupt.train', voc, 0, 1)
# _,_,_,_,_,Y_lex_test,_,_,_,_,_= prep.file2ts(LANG+'/test.blind.cupt.lextrain', voc, 0, 1)
# _,_,_,_,_,Y_truth_test,_,_,_,_,_ = prep.file2ts(LANG+'/test.cupt', voc, 0, 1)



	# logger.log({})

	if wnb:
		wandb.init(project="merger", config=config)
		# print('wandb init')


	# device = 'cuda'
	device = 'cpu'
	# model = Merger3(4, frozen, device).to(device)

	# sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.system.cupt', voc, 0, 1)

	# for s in sentences.tolist():
	# 	for w in s:
	# 		voc.get(w)

	# for s in test_sentences.tolist():
	# 	for w in s:
	# 		voc.get(w)
	# print(len(voc.dic))
	# model = Merger_emb(voc, 4, frozen, device).to(device)

	model = Merger_bert2(4, frozen, device).to(device)
	# model = Merger_mtlb(1, frozen, device).to(device)
	if load:
		try:
			model.load_state_dict(torch.load(load))
		except:
			print("model didn't load, continuing with random weights")




	# tmp = rng.permutation(np.array(data.index))
	# train_id = tmp[:len(tmp)//4*3]
	# test_id = tmp[len(tmp)//4*3:]


	# train_data = data.iloc[train_id]
	# test_data = data.iloc[test_id]

	# train_truth = truth.iloc[train_id]
	# test_truth = truth.iloc[test_id]

	# train_sentences = sentences.iloc[train_id]
	# test_sentences = sentences.iloc[test_id]

	optimizer = optim.Adam(
		model.parameters(),
		lr=lr,
		weight_decay=wd
	)
	bce = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pw))
	model.train()

	def rgd_e(loss, temp, reduce=True):
			# alpha >0.
		loss = loss.unsqueeze(-1)
		out = loss * torch.exp(torch.clamp(loss.detach(), max=temp) / (temp + 1))
		out = out.sum() / len(out) if reduce else out
		return out.squeeze(-1)



	if train:
		if mode == 'normal':
			sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.system.cupt', voc, 0, 1)
			_,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.blind.cupt.lex', voc, 0, 1)

		if mode == 'lex':
			sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.blind.cupt.lex', voc, 0, 1)
			_,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.blind.cupt.lex', voc, 0, 1)

		if mode == 'mtlb':
			sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.system.cupt', voc, 0, 1)
			_,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.system.cupt', voc, 0, 1)
		_,_,_,_,_,Y_truth,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.cupt', voc, 0, 1)

		# sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.system.cupt', voc, 0, 1)
		# _,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.blind.cupt.lex', voc, 0, 1)
		# _,_,_,_,_,Y_truth,_,_,_,_,_ = prep.file2ts(LANG+'/dev1.cupt', voc, 0, 1)

		# sentences,_,_,_,_,Y_system,_,_,_,_,_ = prep.file2ts(LANG+'/dev.system.cupt', voc, 0, 1)
		# _,_,_,_,_,Y_lex,_,_,_,_,_ = prep.file2ts(LANG+'/dev.blind.cupt.lex', voc, 0, 1)
		# _,_,_,_,_,Y_truth,_,_,_,_,_ = prep.file2ts(LANG+'/dev.cupt', voc, 0, 1)
		truth, data = f(Y_system, Y_lex, Y_truth)
		# test_truth, test_data = f(Y_system_test, Y_lex_test, Y_truth_test)

		train_data = data
		train_truth = truth
		train_sentences = sentences
		rng = np.random.default_rng()
		for epoch in range(EPOCH):
			tp =0
			x = 0
			y = 0
			z = 0
			rloss = 0
			# tmp = rng.permutation(np.array(train_data.index))
			tmp = rng.permutation(np.arange(len(train_data)))
			for i in range(len(tmp)//bs+1):
				batch_ids = tmp[i*bs:i*bs+bs]
				if len(batch_ids) == 0:
					continue
				batch = train_data.iloc[batch_ids]
				batch_truth = torch.tensor(train_truth.iloc[batch_ids].values.astype(int)).to(device)
				batch_sentence = train_sentences.loc[batch['sentence_id']]
				
			# print(tmp[i*10:i*10+10])
				# print(B)
				model.train()
				# model.eval()
				optimizer.zero_grad()


				r = merger_preprocessing(
					batch_sentence, batch, model, device
				)
				# a, b = torch.split(
				# 	torch.tensor(
				# 		batch[[0, '_merge']].applymap(list).values.tolist(), device=device
				# 	).transpose(-1, -2), split_size_or_sections=1, dim=-1
				# )
				# r = model(
				# 	# sentences[batch['sentence_id']],
				# 	batch_sentence,
				# 	# a,
				# 	# a.bool().float()
				# 	(a.bool() & ~b.bool()).float(),
				# 	(~a.bool() & b.bool()).float(),
				# 	(a & b).float(),
				# 	(a | b).float()
				# )
				loss = bce(r, batch_truth.float())
				# loss = rgd_e(loss, 10.0)

				loss.backward()
				optimizer.step()
				torch.cuda.empty_cache()
				rloss += loss.item()
				tp += sum((r > 0).int() & batch_truth)
				x += sum((r > 0).int())
				z += sum(batch_truth)

			y = len(Y_truth[lambda x: x.apply(lambda x: 1 in x)])
			p = (tp / x).item() if x.item != 0 else 0
			r = (tp / y).item()
			if wnb:
				wandb.log({
					'loss': rloss,
					'p': p,
					'r': r,
					'f': (2 * p * r) / (p + r) if r+p != 0 else 0,
					'tp': tp,
					'predicted': x,
					'to_predct': y,
					'max_r': z / y,
					'max_f': (2 * (z / y)) / ((z / y) + 1)
				})
			
			if ceval:
				# model.requires_grad_(False)
				# optimizer.zero_grad(set_to_none=True)
				eval_model(model, test_sentences)
				# model.requires_grad_(True)
				# model.bert.partial_freeze(frozen)
			# eval_model(model, test_sentences, test_data, test_truth)

			# print(p, r, x.item(), y, z.item(), (2 * p * r) / (p + r))
			# print()

		if save:
			torch.save(model.state_dict(), save)


	if eval:
		eval_model(model, test_sentences)
		
	
	# # torch.cuda.empty_cache()
	# model.eval()
	# truth_test, data_test = f(Y_system_test, Y_lex_test, Y_truth_test)
	# score= 0
	# tmp = np.array(data_test.index)
	# tp =0
	# x = 0
	# y = 0
	# z = 0
	# rloss = 0
	# # batch_size = 100
	# for i in range(len(tmp)//bs+1):
	# 	batch_ids = tmp[i*bs:i*bs+bs]
	# 	batch_test = data_test.iloc[batch_ids]
	# 	batch_truth_test = torch.tensor(truth_test.iloc[batch_ids].values.astype(int)).to(device)

	# 	a, b = torch.split(
	# 		torch.tensor(
	# 			batch_test[[0, '_merge']].applymap(list).values.tolist(), device=device
	# 		).transpose(-1, -2), split_size_or_sections=1, dim=-1
	# 	)

	# 	r_test = model(
	# 		sentences_test[batch_test['sentence_id']],
	# 		(a.bool() & ~b.bool()).float(),
	# 		(~a.bool() & b.bool()).float(),
	# 		(a & b).float(),
	# 		(a | b).float()
	# 	)

	# 	# r_test = model(
	# 	# 	sentences_test[batch_test['sentence_id']],
	# 	# 	*torch.split(
	# 	# 		torch.tensor(
	# 	# 			batch_test[[0, '_merge']].applymap(list).values.tolist(), device=device
	# 	# 		).transpose(-1, -2), split_size_or_sections=1, dim=-1
	# 	# 	)
	# 	# )
	# 	loss = bce(r_test, batch_truth_test.float())	
	# 	rloss += loss.item()
	# 	tp += sum((r_test > 0).int() & batch_truth_test)
	# 	x += sum((r_test > 0).int())
	# 	z += sum(batch_truth_test)
	# 	# y += sum(batch_truth_test.int())

	# 	# annotated = data_test.iloc[batch_ids].loc[[x for x, y in zip(data_test.iloc[batch_ids].index, r_test.cpu() > 0) if y]]
	# 	# print(tp.item(), x.item(), z.item())
	# # print()
	# # y = len(truth_test.loc[lambda x: x])
	# y = len(Y_truth_test[lambda x: x.apply(lambda x: 1 in x)])
	# p = (tp / x).item()
	# r = (tp / y).item()
	# wandb.log({
	# 	'loss': rloss,
	# 	'p': p,
	# 	'r': r,
	# 	'f': (2 * p * r) / (p + r),
	# 	'tp': tp,
	# 	'predicted': x,
	# 	'to_predct': y,
	# 	'max_r': z / y,
	# 	'max_f': (2 * (z / y)) / ((z / y) + 1)
	# })
	# print(p, r, x.item(), y, z.item(), (2 * p * r) / (p + r))
