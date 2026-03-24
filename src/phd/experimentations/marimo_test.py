import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from phd import liai
    import torch


    LANG = 'Multi'
    DEVorTEST = 'test'
    frozen = False

    parseme_path = '../data/parseme'
    liai_path = '../data/liai'


    print('Loading data...')
    voc = liai.prep.Voc()

    return DEVorTEST, LANG, liai, parseme_path, voc


@app.cell
def _():
    some_var = 2
    return


@app.cell
def _(DEVorTEST, LANG, liai, parseme_path, voc):
    print(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.system.cupt')
    test_sentences,__,__,__,__,Y_system_test,__,__,__,__,__ = liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.system.cupt', voc, 0, 1)
    print(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.blind.cupt.lex')
    __,__,__,__,__,Y_lex_test,__,__,__,__,__= liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.blind.cupt.lex', voc, 0, 1)
    print(f'loading {parseme_path}/1.2/{LANG}/{DEVorTEST}.cupt')
    __,__,__,__,__,Y_truth_test,__,__,__,__,__ = liai.prep.file2ts(f'{parseme_path}/1.2/{LANG}/{DEVorTEST}.cupt', voc, 0, 1)
    return (Y_system_test,)


@app.cell
def _(Y_system_test):
    Y_system_test
    return


if __name__ == "__main__":
    app.run()
