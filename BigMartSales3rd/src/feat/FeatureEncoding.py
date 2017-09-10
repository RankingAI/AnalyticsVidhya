import pandas as pd
import numpy as np
import numba

class FeatureEncoding:
    """"""
    _l_cate_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Establishment_Year',
                  'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    @classmethod
    def ordinal(cls, data, d_values, mode = 'simple'):
        """"""
        if(mode == 'simple'):
            data = cls.__SimpleEncode(data, d_values)
        elif(mode == 'onehot'):
            data = cls.__OneHotEncode(data, d_values)
        elif(mode == 'likelihood'):
            data = cls.__LikelihoodEncode(data, d_values)
        elif(mode == 'mixed'):
            cls._l_cate_cols = [col for col in cls._l_cate_cols if col not in d_values[1].keys()]
            print(cls._l_cate_cols)
            data = cls.__OneHotEncode(data, d_values[0])
            data = cls.__LikelihoodEncode(data, d_values[1])

        return  data

    @classmethod
    def __LikelihoodEncode(cls, data, d_values):
        """"""
        for col in d_values:
            for val in d_values[col]:
                data.loc[data[col] == val, col] = d_values[col][val]
            data[col] = data[col].values.astype(np.float64)
        return  data

    @classmethod
    def __OneHotEncode(cls, data, d_values):
        """"""
        d_values = dict((k, v) for (v, k) in enumerate(list(d_values.keys()), start = 0))
        headers = [v[0] for v in sorted(d_values.items(), key=lambda x: x[1])]
        ##
        ohe = cls.__ApplyOHE(data, d_values)
        ##
        df_ohe = pd.DataFrame(ohe, index = data.index, columns= headers)
        data = pd.concat([data, df_ohe], axis=1)
        data.drop(cls._l_cate_cols, axis = 1, inplace = True)

        return data

    @classmethod
    def __SimpleEncode(cls, data, d_values):
        """"""
        for col in cls._l_cate_cols:
            ##
            en = cls.__ApplySE(data[col].values, d_values[col])
            ##
            df_en = pd.DataFrame(en, index= data.index, columns= ['%snew' % col])
            data = pd.concat([data, df_en], axis=1)
            data[col] = data['%snew' % col]
            data.drop('%snew' % col, axis = 1, inplace = True)

        return data

    @classmethod
    @numba.jit
    def __ApplyLHE(cls, data, d_values):
        """"""

    @classmethod
    @numba.jit
    def __ApplyOHE(cls, data, d_values):
        """"""
        n = len(data)
        result = np.zeros((n, len(d_values)), dtype= 'int8')
        for i in range(n):
            for col in cls._l_cate_cols:
                v = data.ix[i, col]
                if(pd.isnull(v)):
                    result[i, d_values['%s:missing' % col]] = 1
                else:
                    result[i, d_values['%s:%s' % (col, v)]] = 1

        return result

    @classmethod
    @numba.jit
    def __ApplySE(cls, data, ColumnValues):
        """"""
        n = len(data)
        result = np.zeros((n, 1), dtype= 'int8')
        for i in range(n):
            v = data[i]
            if(pd.isnull(v)):
                result[i] = len(ColumnValues)
            else:
                result[i] = ColumnValues[v]

        return result
