
# %%
# %load_ext autoreload
# %autoreload 2
# import importlib
# import phd.liai.core as liai_core
# importlib.reload(liai_core)
# import phd.liai as liai  # re-runµthe star import
# importlib.reload(liai)
# %%
from phd import liai, utils, cupt_parser
# from phd import model_copy, merger
import torch
import rich.console



LANG = 'FR'
DEVorTEST = 'test'
frozen = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parseme_path = '../data/parseme'
liai_path = '../data/liai'

console = rich.console.Console()

# %%
voc = liai.prep.Voc()
with console.status('loading data') as spinner:
	spinner.update(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.mtlb_trained_on_traindev.cupt')
# print(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.system.cupt')
	test_sentences, Y_system_test = (tmp_tuple:=liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.mtlb_trained_on_traindev.cupt', voc, 0, 1))[0], tmp_tuple[5]

	spinner.update(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.lex_lem_dep_css.cupt')
	Y_lex_test= liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.lex_lem_dep_css.cupt', voc, 0, 1)[5]

	spinner.update(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.cupt')
	Y_truth_test = liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.cupt', voc, 0, 1)[5]

#%%
from pathlib import Path
with console.status('loading model') as spinner:
	bert_custom_cache = str(Path('~/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/fdfce55e83dbed325647a63e7e1f5de19f0382ba').expanduser().resolve())
	# bert_custom_cache = None

	model = liai.Merger_with_padding_mask(4, frozen, device, bert_custom_cache=bert_custom_cache).to(device)
	# model = liai.Merger_with_padding_mask(4, frozen, device, bert_custom_cache=Path('~/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/3f076fdb1ab68d5b2880cb87a0886f315b8146f8/')).to(device)
	# model = liai.Merger_with_padding_mask(4, frozen, device, bert_custom_cache='~/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/fdfce55e83dbed325647a63e7e1f5de19f0382ba').to(device)



	state_dict = torch.load(
		f'{liai_path}/45_merger_masked.model',
		map_location=device
	)
	# position_ids was a persistent buffer in older transformers versions; it is
	# non-persistent in newer ones, so we drop it to allow strict loading.
	try:
		# for older versions of transformers
		model.load_state_dict(state_dict)
	except:
		# for newer versions of transformers
		state_dict.pop('bert.embedding.embeddings.position_ids', None)
		model.load_state_dict(state_dict)

#%%
# with console.status('evaluating model') as spinner:
# spinner.update('setting up test data')
df_test = cupt_parser.setup_data_noTT(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.cupt')
# spinner.update('building candidate table and labels')
truth_test, data_test = liai.build_candidate_table_and_labels(Y_system_test, Y_lex_test, Y_truth_test)
# spinner.update('evaluating model')
#%% 
liai.eval_model(model, test_sentences, Y_truth_test, data_test, truth_test, df_test, device,100)

# %%
