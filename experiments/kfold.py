from sklearn.model_selection import KFold


def split(df):
    n = len(df)
    return df.iloc[:n*4//5], df.iloc[n*4//5:]


def kfold(df):
    model = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in model.split(df):
        yield df.iloc[train_index], df.iloc[test_index]
