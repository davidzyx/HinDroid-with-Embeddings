import pandas as pd


class FeatureBuilder():

    def __init__(self, agg_df, labels):
        self.df = agg_df
        self.labels = pd.Series(labels).sort_index()
        self.out = pd.DataFrame()
        self.build()

    def _flatten_col_names(df):
        df.columns = ['.'.join(col).strip() for col in df.columns.values]
        return df

    def _simple_aggregations(df):
        out = df.groupby('package').agg({
            'call': 'size',
            'library': 'nunique',
            'code_block_id': ['mean', 'std', 'median', 'max', 'nunique'],
        })
        return FeatureBuilder._flatten_col_names(out)

    def _invoke_counts_by_type(df):
        out = df.groupby('package')['invocation'].value_counts().unstack(fill_value=0)
        out.columns = [col + '.count' for col in out.columns.values]
        return out

    def numerical_features(self):
        features = [
            FeatureBuilder._simple_aggregations(self.df),
            FeatureBuilder._invoke_counts_by_type(self.df)
        ]
        out = pd.concat(features, axis=1)
        return out

    def _top5_library(df):
        out = df.groupby('package')['library'].apply(
            lambda s: s.value_counts().iloc[:5]
        ).unstack(fill_value=0).clip(upper=1)
        out.columns = ['top5.' + col for col in out.columns.values]
        return out

    def categorical_features(self):
        features = [
            FeatureBuilder._top5_library(self.df)
        ]
        out = pd.concat(features, axis=1)
        return out

    def build(self):
        self.out = pd.concat([
            self.numerical_features(),
            self.categorical_features()
        ], axis=1)
