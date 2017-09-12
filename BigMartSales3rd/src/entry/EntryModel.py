import time
from model.ModelTask import ModelTask

if __name__ == '__main__':
    """"""
    DataDir = '/Users/yuanpingzhou/project/workspace/python/AnalyticsVidhya/BigMartSales3rd/data'

    ## load parameters for tunning
    with open('%s/parameters.txt' % DataDir, 'r') as i_file:
        d_params = eval(i_file.read())
    i_file.close()

    input = '%s/feat/FeatureEngineering' % DataDir
    ## stack model
    strategies = [['lgb', 'lr', 'en']]
    S = 1 ## #start layer
    N = 2 ## #total layers+1
    for i in range(S, N):
        if(i == 1):
            InputDir = input ## first layer
        else:
            InputDir = '%s/L%s' % (DataDir, i - 1)
        OutputDir = '%s/L%s' % (DataDir, i)
        mt = ModelTask(i, strategies[i - 1], d_params, InputDir, OutputDir, data_format= 'csv')
        mt.run()
