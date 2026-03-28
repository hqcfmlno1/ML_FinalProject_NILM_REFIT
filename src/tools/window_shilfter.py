import pandas as pd

class WindowShilfter():
    def shift(df, n):
        tmp = pd.DataFrame()
        df_keep = df[['Time', 'Aggregate']]
        df_app = df.drop(columns = ['Time', 'Unix', 'Aggregate']) 
        for i in range(n):
            tmp = pd.concat([tmp, df_keep.shift(i)], axis = 1)
        df_with_nan = pd.concat([tmp, df_app], axis = 1)
        nan_index = df_with_nan.isnull().sum(axis = 1)[df_with_nan.isnull().sum(axis = 1)>0].index
        df = df_with_nan.drop(nan_index)
        return df
    

