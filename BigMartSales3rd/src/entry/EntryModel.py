import time
from model.ModelTask import ModelTask

if __name__ == '__main__':
    """"""
    DataDir = '/Users/yuanpingzhou/project/workspace/python/AnalyticsVidhya/BigMartSales3rd/data'

    ## load parameters for tunning
    with open('%s/parameters.txt' % DataDir, 'r') as i_file:
        d_params = eval(i_file.read())
    i_file.close()

    ## L1
    InputDir = '%s/feat/FeatureEngineering' % DataDir
    strategies = ['lr', 'lgb', 'en']
    for strategy in strategies:
        print('\n ==== Task for %s begins.  ====\n' % strategy)

        start = time.time()
        print(d_params['L1'][strategy])
        ModelTask.run(strategy, d_params['L1'][strategy], InputDir, '%s/L1' % DataDir, data_format= 'csv', metric_pk= False)
        end = time.time()

        print('\n ==== Task for %s done, time consumed %s. ====\n' % (strategy, (end - start)))

    ## L2
    # InputDir = '%s/L1/aggregate' % DataDir
    # strategies = ['en']
    # for strategy in strategies:
    #     print('\n ==== Task for %s begins.  ====\n' % strategy)
    #
    #     start = time.time()
    #     ModelTask.run(strategy, d_params['L2'][strategy], InputDir, '%s/L2' % DataDir, data_format= 'csv', metric_pk= True)
    #     end = time.time()
    #
    #     print('\n ==== Task for %s done, time consumed %s. ====\n' % (strategy, (end - start)))
