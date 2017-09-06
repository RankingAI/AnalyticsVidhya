from feat.FeatureEngineering import FeatureEngineering
from feat.Preprocess import Preprocess

if __name__ == '__main__':
    """"""
    DataDir = '/Users/yuanpingzhou/project/workspace/python/AnalyticsVidhya/BigMartSales3rd/data'

    ## Preprocess
    InputDir = '%s' % DataDir
    OutputDir = '%s/Preprocess' % DataDir

    tasks = ['holdout', 'kfold']
    #Preprocess.run(tasks, InputDir, OutputDir)

    ## Engineering
    InputDir = '%s/Preprocess' % DataDir
    OutputDir = '%s/FeatureEngineering' % DataDir
    #
    tasks = ['MissingValue']
    fe = FeatureEngineering(InputDir, OutputDir)
    fe.run(tasks)
