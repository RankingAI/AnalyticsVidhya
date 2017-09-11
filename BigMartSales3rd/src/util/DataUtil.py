import dill as pickle
import pandas as pd

class DataUtil:
    """"""
    @classmethod
    def load(cls, file, format= 'pkl'):
        """"""
        data = ''
        if(format == 'pkl'):
            with open(file, 'rb') as i_file:
                data = pickle.load(i_file)
            i_file.close()
        elif(format == 'csv'):
            data = pd.read_csv(file)
        else:
            print('\n format error !!!\n')

        return data

    @classmethod
    def save(cls, data, file, format= 'pkl'):
        """"""
        if(format == 'pkl'):
            with open(file, 'wb') as o_file:
                pickle.dump(data, o_file, -1)
            o_file.close()
        elif(format == 'csv'):
            data.to_csv(file, float_format= '%.4f', index= False)
        else:
            print('\n format error !!! \n')

        return
