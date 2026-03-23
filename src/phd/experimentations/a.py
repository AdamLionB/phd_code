import cupt_parser
import lambda_css_lexicon
import lambda_css
import lambda_css_utils
import mwe_lexicon
import oa_indices

# parseme_path = 'data/parseme'

def data2score_pipeline(
	entry_type,
	train_path: str,
	test_path: str
):
	TT_train, df_train = cupt_parser.setup_data(
		train_path
	)

	TT_test, df_test = cupt_parser.setup_data(
		test_path
	)

	# entry_type = lambda_css.LambdaCSS.get_lCSS({'lemma' : True, 'deprel': False})
	# lcss_lex_formalism = lambda_css_lexicon.lexicon_formalism_for_lcss(lcss_cl)
	
	if issubclass(entry_type, lambda_css.LambdaCSS):
		# lcss_lex_formalism = entry_type
		formalism = mwe_lexicon.Lexicon_formalism.concretize(
			entry_type,
			lambda_css_lexicon.extract_lcss_lexicon,
			lambda_css_lexicon.lexicon_matches_in_sentences
		)
		lex = formalism.instantiate(TT_train)
		pred = lex.match(df_test)
	if issubclass(entry_type, mwe_lexicon.Seq_rep):
		
		test_sorted_columns = lambda_css_utils.sort_column(df_test)
		# sort_column(test_sentences)
		formalism = mwe_lexicon.Lexicon_formalism.concretize(
			entry_type,
			mwe_lexicon.extract_pattern_from_data,
			mwe_lexicon.SeqRep_match
		)
		lex = formalism.instantiate(df_train)
		pred = lex.match(df_test, test_sorted_columns)
	# entry_type = mwe_lexicon.Seq_rep.concretize(['lemma'], mwe_lexicon.disc_fs[0])
	
	# seq_lex = seq_lex_formalism.instantiate(df_train)
	# lcss_lex = lcss_lex_formalism.instantiate(TT_train)

	truth = cupt_parser.get_mwes(df_test)

	

	return oa_indices.full_eval(truth, pred)

def data_diversity_score(
	data_path: str
):
	TT_train, df_data = cupt_parser.setup_data(
		data_path
	)

	data = cupt_parser.get_mwes(df_data)
	inline_data = cupt_parser.inline_mwes(data).reset_index()
	return oa_indices.diversity_eval(inline_data)


