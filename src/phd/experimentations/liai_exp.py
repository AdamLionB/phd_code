
# %%
%load_ext autoreload
%autoreload 2
import importlib
import phd.liai.core as liai_core
importlib.reload(liai_core)
import phd.liai as liai  # re-runµthe star import
importlib.reload(liai)
# %%
from phd import liai, utils, cupt_parser
import torch
import rich.console



LANG = 'FR'
DEVorTEST = 'dev'
frozen = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parseme_path = '../data/parseme'
liai_path = '../data/liai'

console = rich.console.Console()

# %%
voc = liai.prep.Voc()
with console.status('loading data') as spinner:
	spinner.update(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.system.cupt')
# print(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.system.cupt')
	test_sentences, Y_system_test = (tmp_tuple:=liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.system.cupt', voc, 0, 1))[0], tmp_tuple[5]
	spinner.update(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.blind.cupt.lex')
	Y_lex_test= liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.blind.cupt.lex', voc, 0, 1)[5]
	spinner.update(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.cupt')
	Y_truth_test = liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.cupt', voc, 0, 1)[5]

#%%
with console.status('loading model') as spinner:
	model = liai.Merger_with_padding_mask(4, frozen, device).to(device)


	state_dict = torch.load(
		f'{liai_path}/45_merger_masked.model',
		map_location=device
	)
	# position_ids was a persistent buffer in older transformers versions; it is
	# non-persistent in newer ones, so we drop it to allow strict loading.
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
liai.eval_model(model, test_sentences, data_test, truth_test, df_test, device,10)

# %%
