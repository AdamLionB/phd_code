#%%
import datetime
from itertools import permutations
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn 
from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()
from scipy.optimize import linear_sum_assignment


class Simple_Wrapper(nn.Module):
	def __init__(self, mod = None):
		super(Simple_Wrapper, self).__init__()
		self.mod = mod
	def forward(self, x):
		if self.mod is not None:
			return self.mod(x)#[:,:size,:]
		return x
	def partial_freeze(self):
		pass


class Bert2(nn.Module):
	def __init__(self, device, shorten=True) -> None:
		super().__init__()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
		self.embedding = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
		self.device = device
		self.shorten = shorten
	def forward(self, sentences):
		tokens = self.tokenizer(
			sentences.tolist(),
			is_split_into_words=True,
			return_tensors='pt',
			padding='longest'
		)
		tokens = {k: v[:,:512].to(self.device) for k, v in tokens.items()}
		# print([v.shape for k, v in tokens.items()])
		emb = self.embedding(**tokens)['last_hidden_state']
		# print(emb.shape)
		# print(emb)
		emb = torch.einsum('ab, abc -> abc', tokens['attention_mask'], emb)
		
		sen_enc_grpby_tok = [[[101]] + [
				self.tokenizer.encode(w, add_special_tokens=False)
				for w in x
			] + [[102]] for x in sentences.tolist()]
		
		# mask = torch.tensor([[n == 0 for w in x for n in range(len(w))] for x in sen_enc_grpby_tok])
		mask = [torch.tensor([(n == 0) and (m != (len(x) - 1)) for m, w in enumerate(x) for n in range(len(w))], device=self.device) for x in sen_enc_grpby_tok]
		mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=1)
		# emb = torch.masked_select(emb, mask.unsqueeze(-1)).view(emb.shape[0], -1, 768)[:,1:-1,:]
		padded_tensor = torch.zeros_like(emb, device=self.device)
		m = torch.masked_select(emb, mask.unsqueeze(-1)[:,:512,:])
		try:
			padded_tensor[(torch.arange(emb.shape[1]).expand(emb.shape[0], emb.shape[1]) < torch.tensor([sum(x.float()) for x in mask]).unsqueeze(1)).unsqueeze(-1).expand(emb.shape[0], emb.shape[1], emb.shape[-1])] = m
			max_size = max(map(len, sen_enc_grpby_tok)) - 2
			return padded_tensor[:,1:max_size+1,:]
		
		except:
			# print('too long sentence, entire batch weirded out')
			max_size = max(map(len, sen_enc_grpby_tok)) - 2
			return emb[:,1:max_size+1,:]
	def partial_freeze(self, frozen):
		if frozen:
			self.embedding.requires_grad_(False)



class Bert(nn.Module):
	def __init__(self, device, shorten=True) -> None:
		super().__init__()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
		self.embedding = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
		self.device = device
		self.shorten = shorten
	def forward(self, sentences):
		tokens = self.tokenizer(
			sentences.tolist(),
			is_split_into_words=True,
			return_tensors='pt',
			padding='longest'
		)
		tokens = {k: v.to(self.device) for k, v in tokens.items()}
		emb = self.embedding(**tokens)['last_hidden_state']
		emb = torch.einsum('ab, abc -> abc', tokens['attention_mask'], emb)
		
		sen_enc_grpby_tok = [[[101]] + [
				self.tokenizer.encode(w, add_special_tokens=False)
				for w in x
			] + [[102]] for x in sentences.tolist()]
		max_size = max(map(len, sen_enc_grpby_tok)) - 2
		sen_enc_ids = torch.nn.utils.rnn.pad_sequence(
			[
				torch.tensor([n for n, x in enumerate(z) for y in x][:-1] + [0], device=self.device)
				for z in sen_enc_grpby_tok
			],
			batch_first=True,
			padding_value=0
		)
		# print(size, s)

		x = torch.zeros(emb.shape[0],emb.shape[1] ,emb.shape[2], device=self.device).scatter_add_(
			1,
			sen_enc_ids.unsqueeze(dim=-1).expand(emb.shape[0], emb.shape[1] ,emb.shape[2]),
			emb
		)
		y = torch.zeros(emb.shape[0],emb.shape[1] ,emb.shape[2], device=self.device).scatter_add_(
			1,
			sen_enc_ids.unsqueeze(dim=-1).expand(emb.shape[0], emb.shape[1] ,emb.shape[2]),
			# emb / emb
			torch.ones_like(emb)
		)

		if self.shorten:
			x = x[:,1:max_size+1,:]
			y = y[:,1:max_size+1,:]
		return (x / y).nan_to_num(0)
	def partial_freeze(self, frozen):
		if frozen:
			self.embedding.requires_grad_(False)

class AGG(nn.Module):
	def __init__(self, embeddings, func) -> None:
		super(AGG, self).__init__()
		# self.modules = nn.ModuleList()
		self.mods = nn.ModuleList([k for k, _ in embeddings])
		self.srcs = [v for _, v in embeddings]
		self.func = func
	def forward(self, x):
		x = [
					mod(x) if len(srcs) != 1 else mod(x[srcs[0]])
					for mod, srcs in zip(self.mods, self.srcs)]
		# print([y.shape for y in x])
		return self.func(
				x
				# [emb(xx, size) for emb, xx in zip(self.embeddings, x)]
		)
	def partial_freeze(self, frozen):
		for x in self.mods:
			x.partial_freeze(frozen)
	
			
class MWE_Identifier(nn.Module):
	@staticmethod
	def factory(device, conf, max_len, frozen):
		def configs(n):
			if n == 'lem':
				return AGG([
					[Simple_Wrapper(nn.Embedding(32_000, 768)), ['lemme']],
				], sum)
			if n == 'bert':
				return AGG([
					[Bert(device), ['sentence']],
				], sum)
			if n == 'lem+penc':
				return AGG([
					[Simple_Wrapper(nn.Embedding(32_000, 768)), ['lemme']],
					[Simple_Wrapper(pos_encoding(768, device)), ['pos']]
				], sum)
			if n == 'bert+denc':
				return AGG([
					[Bert(device), ['sentence']],
					[Simple_Wrapper(pos_encoding(768, device)), ['deps']]
				], sum)
			if n == 'bert+demb':
				return AGG([
					[Bert(device), ['sentence']],
					[Simple_Wrapper(nn.Embedding(300, 768, device=device)), ['deps']]
				], sum)
			if n == 'bert+penc':
				return AGG([
					[Bert(device), ['sentence']],
					[Simple_Wrapper(pos_encoding(768, device)), ['pos']]
				], sum)
			if n == 'bert+pemb':
				return AGG([
					[Bert(device), ['sentence']],
					[Simple_Wrapper(nn.Embedding(300, 768, device=device)), ['pos']]
				], sum)
			if n == 'bert+penc+denc': #sum deps enc
				return AGG([
					[Bert(device), ['sentence']],
					[Simple_Wrapper(pos_encoding(768, device)), ['pos']],
					[Simple_Wrapper(pos_encoding(768, device)), ['deps']]
				], sum)
			if n == 'bert+penc+denc_alt': # sum deps enc but weird
				return AGG([
					[Bert(device), ['sentence']],
					[AGG([
						[Simple_Wrapper(pos_encoding(768, device)), ['pos']],
						[Simple_Wrapper(pos_encoding(768, device)), ['deps']]
					], sum), ['pos', 'deps']
				]], sum)
			if n == 'bert+penc+demb': # sum deps emb
				return AGG([
					[Bert(device), ['sentence']],
					[Simple_Wrapper(pos_encoding(768, device)), ['pos']],
					[Simple_Wrapper(nn.Embedding(300, 768, device=device)), ['deps']]
				], sum)
			if n == 'bert+{penc_denc}': #cat deps pos enc
				return AGG([
					[Bert(device), ['sentence']],
					[AGG([
						[Simple_Wrapper(pos_encoding(384, device)), ['pos']],
						[Simple_Wrapper(pos_encoding(384, device)), ['deps']]
					], lambda x: torch.cat(x, dim=-1)), ['pos', 'deps']
				]], sum)
			if n == 'lemme+{penc_denc}': #cat deps pos enc
				return AGG([
					[Simple_Wrapper(nn.Embedding(32_000, 768)), ['lemme']],
					[AGG([
						[Simple_Wrapper(pos_encoding(384, device)), ['pos']],
						[Simple_Wrapper(pos_encoding(384, device)), ['deps']]
					], lambda x: torch.cat(x, dim=-1)), ['pos', 'deps']
				]], sum)

		return MWE_Identifier(
			configs(conf),
			device,
			memory_size = 32,
			emb_size = 768,
			sentence_size = 115,
			frozen = frozen
		)

	def __init__(self, embeddings, device, memory_size = 32, emb_size = 768, sentence_size = 115, frozen = False):
		super(MWE_Identifier, self).__init__()
		BERT_emb_size = 768
		self.frozen = frozen
		self.device = device
		self.sentence_size = sentence_size
		self.memory_size = memory_size

		# self.embeddings = embeddings
		

		# self.Tencorder = nn.TransformerEncoder(
		# 	nn.TransformerEncoderLayer(
		# 		emb_size + memory_size,
		# 		16,
		# 		batch_first = True
		# 	),
		# 	2
		# )
		# self.outer = nn.Linear(
		# 	emb_size + memory_size,
		# 	1 + memory_size
		# )
		
		# self.wrapper = Rec_wrapper(
		# 	embeddings,
		# 	nn.Sequential(
		# 		nn.TransformerEncoder(
		# 			nn.TransformerEncoderLayer(
		# 				emb_size + memory_size,
		# 				16,
		# 				batch_first = True
		# 			),
		# 			2
		# 		),
		# 		nn.Linear(
		# 			emb_size + memory_size,
		# 			1 + memory_size
		# 		)
		# 	),
		# 	Simple_Wrapper(),
		# 	mem_cat_dim = -1,
		# 	rec = True,
		# 	device=device
		# )
		# self.wrapper = Rec_wrapper(
		# 	nn.Sequential(
		# 		embeddings,
		# 		nn.TransformerEncoder(
		# 			nn.TransformerEncoderLayer(
		# 				emb_size,
		# 				16,
		# 				batch_first = True
		# 			),
		# 			2
		# 		)
		# 	),
		# 	nn.Linear(
		# 		emb_size + memory_size,
		# 		1 + memory_size
		# 	),
		# 	Simple_Wrapper(),
		# 	mem_cat_dim = -1,
		# 	rec = True,
		# 	device=device
		# )
		self.wrapper = Rec_wrapper(
			embeddings,
			# nn.Sequential(
			# 	embeddings,
			# 	nn.TransformerEncoder(
			# 		nn.TransformerEncoderLayer(
			# 			emb_size,
			# 			16,
			# 			batch_first = True
			# 		),
			# 		2
			# 	)
			# ),
			Simple_Wrapper(),
			# nn.Linear(
			# 	emb_size,
			# 	emb_size
			# ),
			Simple_Wrapper(),
			# nn.Sequential(
			# 	nn.Linear(emb_size, 1)
			# ),
			mem_cat_dim = -1,
			rec = False,
			device=device
		)


		# self.wrapper = Rec_wrapper(
		# 	embeddings,
		# 	# nn.TransformerEncoder(
		# 	# 	nn.TransformerEncoderLayer(
		# 	# 		emb_size + memory_size,
		# 	# 		16,
		# 	# 		batch_first = True
		# 	# 	),
		# 	# 	2
		# 	# ),
		# 	nn.Linear(
		# 		emb_size + memory_size,
		# 		1 + memory_size
		# 	),
		# 	Simple_Wrapper(),
		# 	-1,
		# 	device,
		# 	rec = True
		# )


		# self.wrapper = Rec_wrapper(
		# 	embeddings,
		# 	# nn.TransformerEncoder(
		# 	# 	nn.TransformerEncoderLayer(
		# 	# 		emb_size + memory_size,
		# 	# 		16,
		# 	# 		batch_first = True
		# 	# 	),
		# 	# 	2
		# 	# ),
		# 	Simple_Wrapper(),
		# 	# nn.Linear(
		# 	# 	emb_size,
		# 	# 	emb_size
		# 	# ),
		# 	# nn.Linear(
		# 	# 	20,
		# 	# 	20
		# 	# ),
		# 	Simple_Wrapper(),
		# 	mem_cat_dim = -1,
		# 	device= device,
		# 	rec = False
		# )
	def partial_freeze(self):
		self.wrapper.partial_freeze(self.frozen)
	def forward2(self, input, past_memory, *past_out):
		size = past_memory.shape[1]
		emb = self.embeddings(input)
		enc_src = self.Tencorder(
			torch.cat([emb, past_memory],
			dim=-1)
		)
		out, memory = torch.split(
			self.outer(enc_src),
			[1, self.memory_size],
			dim=-1
		)
		return (memory,) + past_out + (out,)
	def forward(self, *args, **kwargs):
		return self.wrapper(*args, **kwargs)
	

class Norec_Mechanism(nn.Module):
	def __init__(self, mechanism, device) -> None:
		super(Norec_Mechanism, self).__init__()
		self.mechanism = mechanism
		self.device = device
		if 'posenc' in self.mechanism:
			self.pos_enc = pos_encoding2(768, device)
		if 'cenc' in self.mechanism:
			self.c_enc = pos_encoding2(768, device) # TODO replace this
		if 'learnedenc' in self.mechanism:
			self.learned_enc = nn.Sequential(
				nn.Linear(1, 768),
				Simple_Wrapper(torch.sin),
				nn.Linear(768, 768)
			)
			self.l6[-1].weight.data = torch.eye(768)
		if 'bnek' in self.mechanism:
			self.bnek = nn.Sequential(
				nn.Linear(768, 76),
				nn.LeakyReLU(),
				nn.Linear(76, 1)
			)
		if 'fixed' in self.mechanism:
			self.fixed = nn.Sequential(
				nn.Linear(768, 1)
			)
		if 'pred_k' in self.mechanism:
			self.predk = nn.Linear(768, 1)

	def forward(self, x):
		if 'pred_k' in self.mechanism:
			k = self.predk(x).squeeze(dim=-1).mean(dim=-1)
		else:
			k = None
		if 'posenc' in self.mechanism:
			key = self.pos_enc(torch.randn(x.shape[0], 20).to(self.device)).transpose(-1, -2)
		elif 'cenc' in self.mechanism:
			key = self.c_enc(torch.randn(x.shape[0], 20).to(self.device)).transpose(-1, -2)
		elif 'learnedenc' in self.mechanism:
			key = self.learned_enc(
				(torch.arange(0, 20, device=self.device).float().view(-1, 1))
			).transpose(-1, -2).repeat(x.shape[0], 1, 1)
		if 'bnek' in self.mechanism:
			res = self.bnek(torch.einsum('bne, bek -> bnek', x, key).transpose(-1, -2)).squeeze(-1)
		elif 'prod' in self.mechanism:
			res = x @ key
		if 'fixed':
			res = self.fixed(x)
		
		return k, res


class Rec_wrapper(nn.Module):
	def __init__(
		self,
		pre_wrap,
		wrap,
		post_wrap,
		mem_cat_dim,
		device,
		rec=True,
		mechanism='bnek'
	) -> None:
		super(Rec_wrapper, self).__init__()
		self.pre_wrapped = pre_wrap
		self.wrapped = wrap
		self.post_wrapped = post_wrap
		self.mem_cat_dim = mem_cat_dim
		self.rec = rec
		self.device = device
		self.norec = Norec_Mechanism('pred_k fixed', device)
		# self.pos_enc = pos_encoding2(768, device)
		# self.l = nn.Linear(768, 1)
		# # self.l2 = nn.Linear(768, 768)
		# self.l6 = nn.Sequential(
		# 	nn.Linear(1, 768),
		# 	Simple_Wrapper(torch.sin),
		# 	nn.Linear(768, 768)
			
		# )
		# self.l6[-1].weight.data = torch.eye(768)
		# self.l5 = nn.Sequential(
		# 	nn.Linear(768, 76),
		# 	nn.LeakyReLU(),
		# 	nn.Linear(76, 1)
		# )

		# self.l7 = nn.Sequential(
		# 	nn.Linear(768, 10)
		# )
		
	def forward(self, x, past_memory, *past_outputs):
		if self.rec:
			tmp = self.pre_wrapped(x)
			tmp = self.wrapped(torch.cat([tmp, past_memory],dim=-1))
			mem_size = past_memory.shape[self.mem_cat_dim]

			tmp, memory = torch.split(
				tmp,
				[tmp.shape[self.mem_cat_dim] - mem_size, mem_size],
				dim=-1
			)
			out = self.post_wrapped(tmp)
			return (memory,) + past_outputs + (out,)
		else:
			
			emb = self.pre_wrapped(x)
			tmp = self.wrapped(emb)
			nb_exp, tmp = self.norec(tmp)

			# nb_exp = self.l(tmp).squeeze(dim=-1).mean(dim=-1)
			
			# # key = self.l6((torch.arange(0, 20, device=self.device).float().view(-1, 1))).transpose(-1, -2).repeat(tmp.shape[0], 1, 1)
			# key = self.pos_enc(torch.randn(tmp.shape[0], 20).to(self.device)).transpose(-1, -2)

			# tmp = self.l5(torch.einsum('bne, bek -> bnek', tmp, key).transpose(-1, -2)).squeeze(-1)
			# tmp = self.l7(tmp)
			
			# torch.sin(self.l3(torch.arange(0, 19))).repeat(tmp.shape[0], 1)

			# key = self.l4(torch.sin(self.l3(torch.arange(0, 20, device=self.device).float().view(-1, 1)))).transpose(-1, -2).repeat(tmp.shape[0], 1, 1)
			# (b, e, k)

			# key = torch.sin(self.l3(torch.arange(0, 20, device=self.device).float().view(-1, 1))).transpose(-1, -2).repeat(tmp.shape[0], 1, 1)
			# tmp = tmp @ self.l2(self.pos_enc(torch.randn(tmp.shape[0], 20).to(self.device))).transpose(-1, -2)
			# tmp = 
			# print(tmp.shape)
			# tmp = self.l5(torch.einsum('abc, acd -> abcd', tmp, key).transpose(-1, -2)).squeeze(-1)
			# tmp = tmp @ key
			res = self.post_wrapped(tmp)
			
			# if not self.training:
			# 	pass
			# 	mask = nb_exp[:, None] < torch.arange(1, tmp.shape[-1]+1, device=self.device)
			# 	# masked_tensor = tensor.masked_fill(mask, 0)

			# 	res = tmp.masked_fill(mask.repeat(tmp.shape[1], 1, 1).transpose(0, 1), -1)

			
			return (nb_exp, res)
	def partial_freeze(self, frozen):
		try: self.pre_wrapped.partial_freeze(frozen)
		except: pass
		try: self.wrapped.partial_freeze(frozen)
		except: pass
		try: self.post_wrapped.partial_freeze(frozen)
		except: pass
	# def annotate(self, *args, **kwargs):
	# 	if self.rec:
	# 		return self.annotate_rec(*args, **kwargs)
	# 	return self.annotate_simple(*args, **kwargs)
	def annotate(self, x, s, y, lf, mask, pw):
		pass
	


def annotate2(self, x, s, y, lf, mask, pw):
	loss = 0
	right_token = 0
	right_mwe = 0
	on_token = 0
	shoudlbe_on_token = 0
	max_val = 0.0
	n_mwe = 0
	on_mwe = 0
	on_mwe2 = 0
	on_mwe3 = 0
	right_mwe2 = 0
	tot_mwe2 = 0
	right_mwe3 = 0
	tot_mwe3 = 0
	l1 = nn.L1Loss(reduction='sum')
	for nb_mwe, yy in y.groupby(by=1):
		xx = x[yy['index'].to_list()]
		ss = s[yy.index]

		batch_max_len = ss.apply(len).max()
		memory = torch.zeros(xx.shape[0], batch_max_len , self.memory_size, device=self.device)
		out = tuple()
		
		for i in range(nb_mwe):
			# dual, memory, *out = model(xx[:,:,0], xx[:,:,1], memory, *out)
			
			
			# memory, *out = self(xx[:,:,0], xx[:,:,1], ss, memory, *out)
			# memory, *out = self([ss, xx[:,:,0]], memory, *out)

			memory, *out = self({
					'sentence': ss,
					'pos': xx[:,:batch_max_len,0],
					'lemme': xx[:,:batch_max_len,1],
					'deps': xx[:,:batch_max_len,2]
					}, memory, *out)

		outs = torch.cat(out).reshape(-1, xx.shape[0], batch_max_len) 

		for i in range(len(xx)):
			sentence = outs[:,i,:]
			sentence_sig = torch.sigmoid(outs[:,i,:].detach())
			truth = yy.iloc[i]


			if len(truth[0]) > 1:
				best_truth = min([
					(
						lambda y: (y, l1(sentence_sig[:-1], y))
					)(torch.tensor(prod, device=self.device).view(nb_mwe-1, -1).float()[:,:batch_max_len])#.to(device))
					for prod in permutations(truth[0][:-1], len(truth[0][:-1]))
				],key= lambda x: x[1])[0]

					
				best_truth = torch.cat((
					best_truth,
					torch.tensor([truth[0][-1]], device=self.device).float()[:,:batch_max_len])#.to(self.device))
				)
			else:
				best_truth = torch.tensor([truth[0][0]], device=self.device).float()[:,:batch_max_len]#.to(device)

			min_size = min(sentence.shape[1], mask.shape[1])
			loss += F.binary_cross_entropy_with_logits(
				sentence[:,:min_size],
				best_truth[:,:min_size],
				mask[[i],:min_size],
				pos_weight=torch.ones_like(
					sentence[:,:min_size],
					device=self.device
				) * pw
			)
			right_token += sum(sum((best_truth.int() == sentence_sig.round().int()) & best_truth.int() ))
			right_mwe += int(sum((
				sentence_sig.round().int() == best_truth.int()
			).all(dim=-1)[:-1].int()))

			right_mwe2 += int(sum((
				sentence_sig[1:].round().int() == best_truth[1:].int()
			).all(dim=-1)[:-1].int()))

			tot_mwe2 += max(0, best_truth.shape[0] - 2)

			right_mwe3 += int(sum((
				sentence_sig[2:].round().int() == best_truth[2:].int()
			).all(dim=-1)[:-1].int()))

			tot_mwe3 += max(0, best_truth.shape[0] - 3)

			on_token += sum(sum(sentence_sig.round().int()))
			shoudlbe_on_token += sum(sum(best_truth.int()))
			max_val = max(max_val, sentence_sig.max().item())
			n_mwe += best_truth.shape[0] - 1
			on_mwe += (sentence_sig.round().int().sum(1) > 0).sum()
			on_mwe2 += max(0, (sentence_sig.round().int().sum(1) > 0).sum() - 1)
			on_mwe3 += max(0, (sentence_sig.round().int().sum(1) > 0).sum() - 2)

	return (
			loss,
			right_token,
			right_mwe,
			on_token,
			shoudlbe_on_token,
			max_val,
			n_mwe,
			on_mwe,
			right_mwe2,
			right_mwe3,
			tot_mwe2,
			tot_mwe3,
			on_mwe2,
			on_mwe3,
	)

class HashTensorWrapper():
    def __init__(self, tensor):
        self.tensor = tensor
    def __hash__(self):
        return hash(self.tensor.numpy().tobytes())
    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)

def annotate(self, x, s, y, lf, mask, pw):
	# self.wrapper.annotate(x, s, y, lf, mask, pw)

	# init running scores to default values
	loss = 0
	right_token = 0
	right_mwe = 0
	on_token = 0
	shoudlbe_on_token = 0
	max_val = 0
	n_mwe = 0
	on_mwe = 0
	on_mwe2 = 0
	on_mwe3 = 0
	right_mwe2 = 0
	tot_mwe2 = 0
	right_mwe3 = 0
	tot_mwe3 = 0
	k = 0
	l1 = nn.L1Loss(reduction='sum')
	l2 = nn.MSELoss(reduction='mean')

	# groupsby batch by number of MWE in truth
	for nb_mwe, yy in y.groupby(by=1): #TODO remove for norec
		# if self.training:
		# 	if nb_mwe > 3:
		# 		continue

		# gets sub-batch Xs
		xx = x[yy['index'].to_list()]
		ss = s[yy.index]

		# gets N, the len of longest sentence in sub-batch(for VRAM economy)
		batch_max_len = ss.apply(len).max()

		# setup memory and "previous outputs" used by recursive identifier
		memory = torch.zeros(xx.shape[0], batch_max_len , self.memory_size, device=self.device)
		out = tuple()

		# calls the model foward method once for non-recursive model
		# k+1 times for recursive (k = nb mwe in truth)
		nb_it= nb_mwe if self.wrapper.rec else 1
		for i in range(nb_it):
			memory, *out = self({
					'sentence': ss,
					'pos': xx[:,:batch_max_len,0],
					'lemme': xx[:,:batch_max_len,1],
					'deps': xx[:,:batch_max_len,2]
					}, memory, *out)
			
		# concatenats outputs of the model (does nothing for non-rec models)
		outs = torch.cat(out, dim=-1) # shape: (b, N, k')
		
		# iterates over each sentence of the sub-batch one by one (required for matching)
		for i in range(len(xx)):
			# gets the current sentence truth and model output
			# print(yy.iloc[i])
			truth = yy[0].iloc[i] # shape: (N, k+1)
			# truth = torch.tensor(truth).transpose(-1, -2)
			sentence = outs[i,:,:] # shape: (N, k')
			# print(sentence.shape)
			# if model is non-rec
			if not self.wrapper.rec:
				# puts the truth in a tensor (TODO  this shouldn't be necessary, maybe do it way earlier)
				truth = torch.tensor([tuple(x) for x in truth], device=self.device).transpose(-1, -2)
				truth = truth[:,:-1] #shape: (N, k)
				
				# looks at the size diffrence between k and k' 
				dif_size = truth.shape[-1] - sentence.shape[-1]
				# print(truth.shape, sentence.shape, dif_size)
				# pads shortest with 0s, so that shape is (N, K) for both such that K = max(k, k')
				if dif_size < 0:
					# truth += [0]* -dif_size
					truth = torch.concat([truth, torch.zeros(truth.shape[0],-dif_size, device=self.device)], dim=-1)
					# print(truth.shape)
				elif dif_size > 0:
					sentence = torch.concat([sentence, -torch.ones(sentence.shape[0], dif_size, device=self.device)], dim=-1)
				# gets the k' predicted by the model 
				nb_exp = memory[[i]]
			# gets the sigmoid version of the output (used for matching and score, not for loss)
			sentence_sig = torch.sigmoid(sentence.detach()) # TODO add round here ?


			# if model is non-rec
			if not self.wrapper.rec:
				# finds the permutation of the truth that minimize l1 between truth and prediction
				# print(truth.shape, sentence_sig.shape)
				q = torch.tensor([[(x - y[:sentence_sig.shape[0]]).abs().sum() for x in sentence_sig.T] for y in truth.T])
				_, col_ind = linear_sum_assignment(q)

				best_truth = truth[:,col_ind]

			# if model is rec
			else:
				# if there was at least one MWE to predict
				if len(truth) > 1:
					# finds the permutation of the truth that minimize l1 between truth and prediction
					# the last column of the truth and prediction is ignored for the matching
					# (the recursive model has to find that the last prediction is 0s)
					best_truth = min([
						(
							lambda y: (y, l1(sentence_sig[:-1], y))
						)(torch.stack(prod).transpose(-1, -2).float()[:batch_max_len,:])
						for prod in permutations(truth.transpose(-1, -2), truth.shape[-1])
					],key= lambda x: x[1])[0]

					# the last column of the truth is put back in its place
					best_truth = torch.cat((
						best_truth,
						torch.tensor([truth[-1]], device=self.device).transpose(-1, -2).float()[:batch_max_len,:])
					)

				# if there was no MWE to predict
				else:
					best_truth = torch.tensor([truth[0]], device=self.device).transpose(-1, -2).float()[:batch_max_len,:] #.to(device)
			# gets n, the min between N (longest sentence of sub-batch) and mask longest sentence 
			min_size = min(sentence.shape[0], mask.shape[1]) # TODO useless ? how is it not always N ?

			# print(sentence.shape, truth.shape)

			# computes loss for rec model
			if self.wrapper.rec: #TODO correct ?
				loss += F.binary_cross_entropy_with_logits(
					sentence[:,:min_size],
					best_truth[:,:min_size],
					mask[[i],:min_size].transpose(-1, -2),
					pos_weight=torch.ones_like(
						sentence[:,:min_size],
						device=self.device
					) * pw
				)
			# compute loss for non-rec model
			else:
				if len(truth) > 1:

					# print(nb_mwe)
					# try:
					loss += F.binary_cross_entropy_with_logits(
						sentence[:min_size].float(),
						best_truth[:min_size].float(),
						mask[[i],:min_size].transpose(-1, -2),
						pos_weight=torch.ones_like(
							sentence[:min_size],
							device=self.device
						) * pw
					)  #* 0.5 + 0.5 * l2(nb_exp.float(), torch.tensor([len(truth) - 1]).float().to(self.device))					
					# except:
					# 	print(sentence)
					# 	print(best_truth)
					# 	exit()
				else:
					# pass
					# loss += l2(nb_exp.float(), nb_exp.float().detach())
					loss += 0.5 * l2(nb_exp.float(), torch.zeros(1, device=self.device))
			
			sentence_sig = sentence_sig[:min_size]
			best_truth = best_truth[:min_size]

			# if (best_truth.sum(0) > 0).sum() == 0 and (sentence_sig.round().int() == best_truth.int()).all(dim=0).sum() != 0:
			# 	print(best_truth)
			# 	print(sentence_sig)
			# 	print((best_truth.sum(0) > 0).sum())
			# 	print((sentence_sig.round().int() == best_truth.int()).all(dim=0).sum())
			# 	exit()
			if len(truth) > 1:
				if self.wrapper.rec:
					right_mwe += (
						sentence_sig.round().int() == best_truth.int()
					).all(dim=-2)[:-1].sum()

					right_mwe2 += (
						sentence_sig[:,1:].round().int() == best_truth[:,1:].int()
					).all(dim=-2)[:-1].sum()

					right_mwe3 += (
						sentence_sig[:,2:].round().int() == best_truth[:,2:].int()
					).all(dim=-2)[:-1].sum()
				else:
					right_mwe += (
						(sentence_sig > 0.5) & best_truth.bool()
					).all(dim=0).sum()

					right_mwe2 += (
						(sentence_sig[:,1:] > 0.5) & best_truth[:,1:].bool()
					).all(dim=-2).sum()

					right_mwe3 += (
						(sentence_sig[:,2:] > 0.5) & best_truth[:,2:].bool()
					).all(dim=-2).sum()
				right_token += ((best_truth.int() == sentence_sig.round().int()) & best_truth.int()).sum()
				
				on_token += sentence_sig.round().int().sum()
				on_mwe += (sentence_sig.round().int().sum(0) > 0).sum()
				# print(sentence_sig.shape, best_truth.shape, on_mwe, right_mwe, (best_truth.sum(0) > 0).sum())
				# exit()
				if not self.wrapper.rec:
					k += nb_exp.item()
				max_val = max(max_val, torch.sigmoid(sentence.detach()).max().item())
			
			on_mwe2 += max(0, (sentence_sig.round().int().sum(0) > 0).sum() - 1)
			on_mwe3 += max(0, (sentence_sig.round().int().sum(0) > 0).sum() - 2)
			shoudlbe_on_token += (best_truth.int()).sum()
			if self.wrapper.rec:
				# n_mwe += best_truth.shape[1] - 1
				# tot_mwe2 += max(0, best_truth.shape[1] - 2)
				# tot_mwe3 += max(0, best_truth.shape[1] - 3)
				n_mwe += (best_truth.sum(0) > 0).sum()
				tot_mwe2 += max(0, (best_truth.sum(0) > 0).sum() - 1)
				tot_mwe3 += max(0, (best_truth.sum(0) > 0).sum() - 2)
			else:
				# n_mwe += best_truth.shape[1]
				n_mwe += (best_truth.sum(0) > 0).sum()
				tot_mwe2 += max(0, (best_truth.sum(0) > 0).sum() - 1)
				tot_mwe3 += max(0, (best_truth.sum(0) > 0).sum() - 2)
			

	return (
			loss,
			right_token,
			right_mwe,
			on_token,
			shoudlbe_on_token,
			max_val,
			n_mwe,
			on_mwe,
			right_mwe2,
			right_mwe3,
			tot_mwe2,
			tot_mwe3,
			on_mwe2,
			on_mwe3,
			k
	)


	
def train_model(self,
	data,
	sentences,
	Y_train,
	train_ids,
	optimizer,
	batch_size,
	epochs,
	save,
	logger,
	data_test,
	sentences_test,
	Y_test,
	test_ids,
	pos_weight,
	mask,
	early_stop=0
	):
	Y_train_grps = pd.DataFrame(Y_train.groupby(level=0).apply(list))
	Y_train_grps[1] = Y_train_grps[0].apply(len)
	# a should be between 0.5 and 1, 1 is regular L1 loss, 0.5 only mistake in a direction are taken into account
	cl = lambda a : lambda y, x : (a *  torch.abs(x - y) + (1-a) * (x - y)).sum()
	custom_loss = cl(1)
	bce = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor(pos_weight))
	tot_batch = 0
	scorer = Scorer()
	eval_(self, data_test, sentences_test, Y_test, test_ids, logger, mask, pos_weight)
	
	last_scores = []
	for epoch in range(epochs):
		print(f'epoch={epoch} {datetime.datetime.now()}', flush=True)

		for batch, ids in enumerate(
			torch.utils.data.DataLoader(
				torch.tensor(train_ids).long(),
				batch_size=batch_size, shuffle=True
			)
		):
		
			tot_batch += 1
			b_loss = 0
			x = data[ids]
			s = sentences[list(ids)]
			optimizer.zero_grad()
			y = Y_train_grps.loc[ids].reset_index().reset_index().set_index('sentence_id')
			res = annotate(self, x, s, y, bce, mask, pos_weight)
			scorer.batch(res, tot_batch, logger)
			b_loss = res[0]
			b_loss.backward()
			optimizer.step()

		scorer.epoch(logger)

		if save:
			torch.save(self.state_dict(), save)
		if save:
			if (epoch+1) % 10 == 0:
				torch.save(self.state_dict(), save+'_'+str(epoch+1))
		if data_test != None:
			last_scores.append(eval_(self, data_test, sentences_test, Y_test, test_ids, logger, mask, pos_weight))
		if early_stop != 0:
			if last_scores[-early_stop] < last_scores[-1]:
				return


def eval_(self, data, sentences, Y_test, test_ids, logger, mask, pos_weight):
	self.eval()
	self.requires_grad_(False)
	Y_test_grps = pd.DataFrame(Y_test.groupby(level=0).apply(list))
	Y_test_grps[1] = Y_test_grps[0].apply(len)

	scorer = Scorer()
	bce = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor(1)) #FIXME
	for batch, ids in enumerate(
			torch.utils.data.DataLoader(
				torch.tensor(test_ids).long(),
				batch_size=30, shuffle=True
			)
		):
			
			b_loss = 0
			x = data[ids]
			s = sentences[list(ids)]
			y = Y_test_grps.loc[ids].reset_index().reset_index().set_index('sentence_id')

			scorer.batch(annotate(self, x, s, y, bce, mask, pos_weight), 0, None)

	r = scorer.epoch(logger, True)
	self.train()
	self.requires_grad_(True)
	self.partial_freeze()
	return r


def pos_encoding(n_dim, device = 'cpu'):
	f = lambda p, i : torch.sin(p / (10_000 ** (i/n_dim)))
	g = lambda p, i : torch.cos(p / (10_000 ** (i/n_dim)))
	def intern(x):
		tmp = torch.arange(1, n_dim+1, device=device)
		tmp = tmp.repeat(x.shape[1], x.shape[0], 1).transpose(0, 1)
		tmp[:,:,::2] = tmp[:,:,1::2]-2

		res = x.detach().repeat(n_dim, 1, 1).float().permute(1, 2, 0)

		res[:,:,1::2] = f(
			res[:,:,1::2],
			tmp[:,:,1::2]
		)
		res[:,:,::2] = g(
			res[:,:,::2],
			tmp[:,:,::2]
		)
		return res
	return intern

def pos_encoding2(n_dim, device = 'cpu'):
	f = lambda p, i : torch.sin(p / (10 ** (i/n_dim)))
	g = lambda p, i : torch.cos(p / (10 ** (i/n_dim)))

	# f = lambda p, i : torch.sin((i* p) / torch.pi)
	# g = lambda p, i : torch.cos((i* p) / torch.pi)
	def intern(x):
		tmp = torch.arange(1, n_dim+1, device=device)
		tmp = tmp.repeat(x.shape[1], x.shape[0], 1).transpose(0, 1)
		tmp[:,:,::2] = tmp[:,:,1::2]-2

		res = x.detach().repeat(n_dim, 1, 1).float().permute(1, 2, 0)

		res[:,:,1::2] = f(
			res[:,:,1::2],
			tmp[:,:,1::2]
		)
		res[:,:,::2] = g(
			res[:,:,::2],
			tmp[:,:,::2]
		)
		return res
	return intern
# %%
class CBOW_indexed(nn.Module):
	def __init__(self, vocab_size, embedding_size, sentence_size):
		super(MWE_Identifier, self).__init__()

		self.encoder = nn.Embedding(vocab_size, embedding_size),
		self.f = nn.Linear(embedding_size, vocab_size)
		
	def forward(self, x):
		
		return self.f(torch.sum(self.encoder(x),  dim= -2))
		

# torch.masked_select(emb(W), torch.tensor([[1], [1], [0], [0], [0]]).bool()).view(2, -1)

	
from  collections import defaultdict
class Scorer:
	def __init__(self):
		self.scores = defaultdict(float)

	def batch(self, res, tot_batch, logger):
		# res = self.annotate(x, s, y, bce)
		# print(res[0])
		self.scores['it'] += 1
		self.scores['tot_batch'] = tot_batch
		self.scores['b_loss'] = res[0].detach()
		self.scores['b_tp_token'] = res[1] 
		self.scores['b_tp_mwe'] = res[2]
		self.scores['b_pos_token'] = res[3]
		self.scores['b_true_token'] = res[4]
		self.scores['b_max_val'] = res[5]
		self.scores['b_true_mwe'] = res[6]
		self.scores['b_pos_mwe'] = res[7]
		self.scores['b_tp_mwe2'] = res[8]
		self.scores['b_tp_mwe3'] = res[9]
		self.scores['b_true_mwe2'] = res[10]
		self.scores['b_true_mwe3'] = res[11]
		self.scores['b_pos_mwe2'] = res[12]
		self.scores['b_pos_mwe3'] = res[13]
		try:
			self.scores['b_k'] = res[14]
		except:
			self.scores['b_k'] = 0

		self.scores['b_p_token'] = self.scores['b_tp_token'] / self.scores['b_pos_token'] if self.scores['b_pos_token'] != 0 else 0
		self.scores['b_r_token'] = self.scores['b_tp_token'] / self.scores['b_true_token'] if self.scores['b_true_token'] != 0 else 0
		self.scores['b_f_token'] = (2 * self.scores['b_p_token'] * self.scores['b_r_token']) / (
				self.scores['b_p_token'] + self.scores['b_r_token']) if (self.scores['b_p_token'] + self.scores['b_r_token']) != 0 else 0
		self.scores['b_p_mwe'] = self.scores['b_tp_mwe'] / self.scores['b_pos_mwe'] if self.scores['b_pos_mwe'] != 0 else 0
		self.scores['b_r_mwe'] = self.scores['b_tp_mwe'] / self.scores['b_true_mwe'] if self.scores['b_true_mwe'] != 0 else 0
		self.scores['b_f_mwe'] = (2 * self.scores['b_p_mwe'] * self.scores['b_r_mwe']) / (
				self.scores['b_p_mwe'] + self.scores['b_r_mwe']) if (self.scores['b_p_mwe'] + self.scores['b_r_mwe']) != 0 else 0
		# self.scores['b_a_mwe'] = self.scores['b_tp_mwe'] / self.scores['b_true_mwe'] if self.scores['b_true_mwe'] != 0 else 0
		self.scores['b_r_mwe2'] = self.scores['b_tp_mwe2'] / self.scores['b_true_mwe2'] if self.scores['b_true_mwe2'] != 0 else 0
		self.scores['b_r_mwe3'] = self.scores['b_tp_mwe3'] / self.scores['b_true_mwe3'] if self.scores['b_true_mwe3'] != 0 else 0
		self.scores['b_p_mwe2'] = self.scores['b_tp_mwe2'] / self.scores['b_pos_mwe2'] if self.scores['b_pos_mwe2'] != 0 else 0
		self.scores['b_p_mwe3'] = self.scores['b_tp_mwe3'] / self.scores['b_pos_mwe3'] if self.scores['b_pos_mwe3'] != 0 else 0
		self.scores['b_f_mwe2'] = (2 * self.scores['b_p_mwe2'] * self.scores['b_r_mwe2']) / (
				self.scores['b_p_mwe2'] + self.scores['b_r_mwe2']) if (self.scores['b_p_mwe2'] + self.scores['b_r_mwe2']) != 0 else 0
		self.scores['b_f_mwe3'] = (2 * self.scores['b_p_mwe3'] * self.scores['b_r_mwe3']) / (
				self.scores['b_p_mwe3'] + self.scores['b_r_mwe3']) if (self.scores['b_p_mwe3'] + self.scores['b_r_mwe3']) != 0 else 0
		if logger:
			logger.log({
				'batch': self.scores['tot_batch'],
				'it': self.scores['it'],
				'b_loss': self.scores['b_loss'],
				'b_tp_token': self.scores['b_tp_token'],
				'b_tp_mwe': self.scores['b_tp_mwe'],
				'b_pos_token': self.scores['b_pos_token'],
				'b_true_token': self.scores['b_true_token'],
				'b_max_val': self.scores['b_max_val'],
				'b_true_mwe': self.scores['b_true_mwe'],
				'b_pos_mwe': self.scores['b_pos_mwe'],
				'b_p_token': self.scores['b_p_token'],
				'b_r_token': self.scores['b_r_token'],
				'b_f_token': self.scores['b_f_token'],
				'b_p_mwe': self.scores['b_p_mwe'],
				'b_r_mwe': self.scores['b_r_mwe'],
				'b_f_mwe': self.scores['b_f_mwe'],
				# 'b_a_mwe': self.scores['b_a_mwe'],
				'b_f_mwe2': self.scores['b_f_mwe2'],
				'b_f_mwe3': self.scores['b_f_mwe3'],
				'b_p_mwe2': self.scores['b_p_mwe2'],
				'b_p_mwe3': self.scores['b_p_mwe3'],
				'b_r_mwe2': self.scores['b_r_mwe2'],
				'b_r_mwe3': self.scores['b_r_mwe3'],
				'b_true_mwe2': self.scores['b_true_mwe2'],
				'b_true_mwe3': self.scores['b_true_mwe3'],
				'b_pos_mwe2': self.scores['b_pos_mwe2'],
				'b_pos_mwe3': self.scores['b_pos_mwe3'],
				'b_k': self.scores['b_k']
				})
		
		self.scores['e_loss'] += self.scores['b_loss']
		self.scores['e_tp_token'] += self.scores['b_tp_token']
		self.scores['e_tp_mwe'] += self.scores['b_tp_mwe']
		self.scores['e_pos_token'] += self.scores['b_pos_token']
		self.scores['e_true_token'] += self.scores['b_true_token']
		self.scores['e_max_val'] = max(self.scores['e_max_val'], self.scores['b_max_val'])
		self.scores['e_true_mwe'] += self.scores['b_true_mwe']
		self.scores['e_pos_mwe'] += self.scores['b_pos_mwe']
		self.scores['e_pos_mwe2'] += self.scores['b_pos_mwe2']
		self.scores['e_pos_mwe3'] += self.scores['b_pos_mwe3']
		self.scores['e_tp_mwe2'] += self.scores['b_tp_mwe2']
		self.scores['e_true_mwe2'] += self.scores['b_true_mwe2']
		self.scores['e_tp_mwe3'] += self.scores['b_tp_mwe3']
		self.scores['e_true_mwe3'] += self.scores['b_true_mwe3']
		self.scores['e_k'] += self.scores['b_k']
		# print(res[0])
		# return res[0]
	def epoch(self, logger, eval = False):
		x = 't' if eval else 'e'
		self.scores['e_p_token'] = self.scores['e_tp_token'] / self.scores['e_pos_token'] if self.scores['e_pos_token'] != 0 else 0
		self.scores['e_r_token'] = self.scores['e_tp_token'] / self.scores['e_true_token'] if self.scores['e_true_token'] != 0 else 0
		self.scores['e_f_token'] = (2 * self.scores['e_p_token'] * self.scores['e_r_token']) / (
				self.scores['e_p_token'] + self.scores['e_r_token']) if (self.scores['e_p_token'] + self.scores['e_r_token']) != 0 else 0
		self.scores['e_p_mwe'] = self.scores['e_tp_mwe'] / self.scores['e_pos_mwe'] if self.scores['e_pos_mwe'] != 0 else 0
		self.scores['e_r_mwe'] = self.scores['e_tp_mwe'] / self.scores['e_true_mwe'] if self.scores['e_true_mwe'] != 0 else 0
		self.scores['e_f_mwe'] = (2 * self.scores['e_p_mwe'] * self.scores['e_r_mwe']) / (
				self.scores['e_p_mwe'] + self.scores['e_r_mwe']) if (self.scores['e_p_mwe'] + self.scores['e_r_mwe']) != 0 else 0
		r = self.scores['e_f_mwe']
		# self.scores['e_a_mwe'] = self.scores['e_tp_mwe'] / self.scores['e_true_mwe'] if self.scores['e_true_mwe'] != 0 else 0
		self.scores['e_r_mwe2'] = self.scores['e_tp_mwe2'] / self.scores['e_true_mwe2'] if self.scores['e_true_mwe2'] != 0 else 0
		self.scores['e_r_mwe3'] = self.scores['e_tp_mwe3'] / self.scores['e_true_mwe3'] if self.scores['e_true_mwe3'] != 0 else 0
		
		self.scores['e_p_mwe2'] = self.scores['e_tp_mwe2'] / self.scores['e_pos_mwe2'] if self.scores['e_pos_mwe2'] != 0 else 0
		self.scores['e_p_mwe3'] = self.scores['e_tp_mwe3'] / self.scores['e_pos_mwe3'] if self.scores['e_pos_mwe3'] != 0 else 0
		self.scores['e_f_mwe2'] = (2 * self.scores['e_p_mwe2'] * self.scores['e_r_mwe2']) / (
				self.scores['e_p_mwe2'] + self.scores['e_r_mwe2']) if (self.scores['e_p_mwe2'] + self.scores['e_r_mwe2']) != 0 else 0
		self.scores['e_f_mwe3'] = (2 * self.scores['e_p_mwe3'] * self.scores['e_r_mwe3']) / (
				self.scores['e_p_mwe3'] + self.scores['e_r_mwe3']) if (self.scores['e_p_mwe3'] + self.scores['e_r_mwe3']) != 0 else 0
		if logger :
			logger.log({
				'epoch': self.scores['epoch'],
				x+'_loss': self.scores['e_loss'],
				x+'_tp_token': self.scores['e_tp_token'],
				x+'_tp_mwe': self.scores['e_tp_mwe'],
				x+'_pos_token': self.scores['e_pos_token'],
				x+'_true_token': self.scores['e_true_token'],
				x+'_max_val': self.scores['e_max_val'],
				x+'_true_mwe': self.scores['e_true_mwe'],
				x+'_pos_mwe': self.scores['e_pos_mwe'],
				x+'_p_token': self.scores['e_p_token'],
				x+'_r_token': self.scores['e_r_token'],
				x+'_f_token': self.scores['e_f_token'],
				x+'_p_mwe': self.scores['e_p_mwe'],
				x+'_r_mwe': self.scores['e_r_mwe'],
				x+'_f_mwe': self.scores['e_f_mwe'],
				# x+'_a_mwe': self.scores['e_a_mwe'],
				x+'_f_mwe2': self.scores['e_f_mwe2'],
				x+'_f_mwe3': self.scores['e_f_mwe3'],
				x+'_p_mwe2': self.scores['e_p_mwe2'],
				x+'_p_mwe3': self.scores['e_p_mwe3'],
				x+'_r_mwe2': self.scores['e_r_mwe2'],
				x+'_r_mwe3': self.scores['e_r_mwe3'],
				x+'_true_mwe2': self.scores['e_true_mwe2'],
				x+'_true_mwe3': self.scores['e_true_mwe3'],
				x+'_pos_mwe2': self.scores['e_pos_mwe2'],
				x+'_pos_mwe3': self.scores['e_pos_mwe3'],
				x+'_k': self.scores['e_k'],
				})

		self.scores['e_loss'] = 0
		self.scores['e_tp_token'] = 0
		self.scores['e_tp_mwe'] = 0
		self.scores['e_pos_token'] = 0
		self.scores['e_true_token'] = 0
		self.scores['e_max_val'] = 0
		self.scores['e_true_mwe'] = 0
		self.scores['e_pos_mwe'] = 0
		self.scores['e_pos_mwe2'] = 0
		self.scores['e_pos_mwe3'] = 0
		self.scores['e_tp_mwe2'] = 0
		self.scores['e_true_mwe2'] = 0
		self.scores['e_tp_mwe3'] = 0
		self.scores['e_true_mwe3'] = 0
		self.scores['e_k'] = 0
		return r

