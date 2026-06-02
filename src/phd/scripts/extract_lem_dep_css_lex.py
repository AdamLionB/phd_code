# %%
from phd import lexicon, cupt_parser, lcss_lex,  lambdacss
import sys
import os

LEX_BASE = lambdacss.LambdaCSS_spec({'lemma' : True, 'deprel': False})
ENTRY_TYPE = lcss_lex.LCSSAdapter(LEX_BASE)


def generate_lexicon_from_corpus_TT(TT_corpus) -> lexicon.MWE_lexicon[lambdacss.LambdaCSS[dict[str, bool]]]:
	return lexicon.MWE_lexicon(ENTRY_TYPE.instantiate(TT_corpus), ENTRY_TYPE)


def generate_lexicon_from_corpus_path(corpus_path) -> lexicon.MWE_lexicon[lambdacss.LambdaCSS[dict[str, bool]]]:
	TT_corpus, df_corpus = cupt_parser.setup_data(
		corpus_path
	)
	return generate_lexicon_from_corpus_TT(TT_corpus)

def save_lexicon(lex, json_path):
	lex.to_json(json_path)

def load_lexicon(json_path) -> lexicon.MWE_lexicon[lambdacss.LambdaCSS[dict[str, bool]]]:
	return lexicon.MWE_lexicon.from_json(json_path)

def generate_or_load_lexicon_from_corpus_TT(TT_corpus, json_path) -> lexicon.MWE_lexicon[lambdacss.LambdaCSS[dict[str, bool]]]:
	if os.path.exists(json_path):
		return load_lexicon(json_path)
	else:
		lex = generate_lexicon_from_corpus_TT(TT_corpus)
		save_lexicon(lex, json_path)
		return lex

def generate_or_load_lexicon_from_corpus_path(corpus_path, json_path) -> lexicon.MWE_lexicon[lambdacss.LambdaCSS[dict[str, bool]]]:
	if os.path.exists(json_path):
		return load_lexicon(json_path)
	else:
		lex = generate_lexicon_from_corpus_path(corpus_path)
		save_lexicon(lex, json_path)
		return lex

def main(corpus_path, output_path):
	if os.path.exists(output_path):
		print(f"Output file {output_path} already exists. Exiting.")
		return
	
	lex = generate_lexicon_from_corpus_path(corpus_path)
	save_lexicon(lex, output_path)

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: python extract_lem_dep_css_lex.py <gold .cupt or .dcupt file> <output .json file>")
		sys.exit(1)
	main(sys.argv[1], sys.argv[2])


