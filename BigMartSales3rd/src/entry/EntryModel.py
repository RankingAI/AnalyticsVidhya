import time
import pandas as pd
from model.ModelTask import ModelTask
from util.DataUtil import DataUtil

if __name__ == '__main__':
    """"""
    DataDir = '/Users/yuanpingzhou/project/workspace/python/AnalyticsVidhya/BigMartSales3rd/data'

    ## load parameters for tunning
    with open('%s/parameters.txt' % DataDir, 'r') as i_file:
        d_params = eval(i_file.read())
    i_file.close()

    input = '%s/feat/FeatureEngineering' % DataDir
    ## stack model
    strategies = [['lr', 'en', 'rgf'], ['lgb']]
    #strategies = [['rf', 'rgf', 'lr', 'en', 'lgb'], ['lgb']]
    S = 1 ## #start laer
    N = S + 1 ## #last layer
    M = 'aggregate'
    F = False
    for i in range(S, N):
        if(i == 1):
            InputDir = input ## first layer
        else:
            InputDir = '%s/L%s/%s' % (DataDir, i - 1, 'aggregate')
        OutputDir = '%s/L%s' % (DataDir, i)
        mt = ModelTask(i, strategies[i - 1], d_params, InputDir, OutputDir, data_format= 'csv')
        mt.run(mode= M, full= F)

    ## submit
    # raw_submit = DataUtil.load('%s/test.csv' % DataDir, 'csv')
    # test_submit = DataUtil.load('%s/L%s/aggregate/submit/test.csv' % (DataDir, N - 1), 'csv')
    # result_submit = pd.DataFrame(index= raw_submit.index)
    # result_submit['Item_Identifier'] = raw_submit['Item_Identifier']
    # result_submit['Outlet_Identifier'] = raw_submit['Outlet_Identifier']
    # result_submit['Item_Outlet_Sales'] = test_submit[test_submit.columns[-1]]
    # DataUtil.save(result_submit, '%s/submit.csv' % DataDir, 'csv')
