class MissingValue:
    """"""
    @classmethod
    def impute(cls, data):
        """"""
        ## categorial features
        data.loc[data['Outlet_Size'].isnull(),'Outlet_Size'] = 'missing'

        ## numeric features
        data.loc[data['Item_Weight'].isnull(), 'Item_Weight'] = -1

        return data
