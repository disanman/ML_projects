import re
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline


def get_column_names_from_ColumnTransformer(column_transformer, verbose=True):
    col_names = []
    transformers = column_transformer.named_transformers_   # returns a dict
    for key, transformer in transformers.items():
        print(f'\nAnalyzing transformer: \'{key}\'') if verbose else None
        if isinstance(transformer, Pipeline):
            steps = transformer.steps
            for step_name, step in steps:
                print(f' Step {step_name} of transformer {key}') if verbose else None
                if isinstance(step, DataFrameMapper):
                    print('  DataFrameMapper') if verbose else None
                    col_names = step.transformed_names_.copy()   # careful, hard reset of names! WIP
                    print('  features: ', col_names) if verbose else None
                else:
                    try:
                        cols = step.get_feature_names()
                        col_names.extend(cols)
                        print('  features: ', cols) if verbose else None
                    except AttributeError:
                        pass
    # clean names
    col_names = [re.sub('[^(0-9_a-zA-Z)]', '', col) for col in col_names]
    return col_names
