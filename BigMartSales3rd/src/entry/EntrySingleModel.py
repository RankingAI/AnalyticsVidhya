from model.SingleModel import SingleModel

if __name__ == '__main__':
    """"""
    InputDir = '/Users/yuanpingzhou/project/workspace/python/AnalyticsVidhya/BigMartSales3rd/data/FeatureEngineering'
    OutputDir = '/Users/yuanpingzhou/project/workspace/python/AnalyticsVidhya/BigMartSales3rd/data/SingleModel'

    strategies = ['lgb']
    SingleModel.run(strategies, InputDir, OutputDir)
