from feat.FeatureEngineering import FeatureEngineering
from feat.Preprocess import Preprocess

if __name__ == '__main__':
    """"""
    Input = '/Users/yuanpingzhou/project/workspace/python/AnalyticsVidhya/BigMartSales3rd/data'
    Output = '%s/feat' % Input

    ## Preprocess
    InputDir = '%s' % Input
    OutputDir = '%s/Preprocess' % Output
    tasks = ['holdout', 'kfold']
    Preprocess.run(tasks, InputDir, OutputDir)

    ## Engineering
    InputDir = '%s/Preprocess' % Output
    OutputDir = '%s/FeatureEngineering' % Output
    tasks = ['MissingValue', 'FeatureEncoding']
    fe = FeatureEngineering(InputDir, OutputDir)
    fe.run(tasks, encode_type= 'onehot')
