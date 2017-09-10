import time
import numpy as np
import pandas as pd
import os,sys
import dill as pickle
from sklearn.cross_validation import KFold

class Preprocess:
    """"""
    _HoldoutRatio = 0.1
    _kfold = 5
    ##
    _l_cate_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Establishment_Year',
                  'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    @classmethod
    def __LaunchTask(cls, task, InputDir, OutputDir):
        """"""
        print('----- Task %s begins ...' % task)
        start = time.time()
        if (os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        if(task == 'holdout'):
            """"""
            TrainData = pd.read_csv('%s/train.csv' % InputDir)
            TestData = pd.read_csv('%s/test.csv' % InputDir)

            TrainData['Outlet_Establishment_Year'] = TrainData['Outlet_Establishment_Year'].values.astype(str)
            TestData['Outlet_Establishment_Year'] = TestData['Outlet_Establishment_Year'].values.astype(str)
            d_item_weight = {}
            for i in range(len(TrainData)):
                item_id = TrainData.ix[i, 'Item_Identifier']
                item_weight = TrainData.ix[i, 'Item_Weight']
                if(pd.isnull(item_weight) == False):
                    d_item_weight[item_id] = item_weight
            for i in range(len(TestData)):
                item_id = TestData.ix[i, 'Item_Identifier']
                item_weight = TestData.ix[i, 'Item_Weight']
                if(pd.isnull(item_weight) == False):
                    d_item_weight[item_id] = item_weight
            item_id_values = TrainData.ix[TrainData['Item_Weight'].isnull() == True,'Item_Identifier'].value_counts().index.values
            for iv in item_id_values:
                TrainData.loc[TrainData['Item_Identifier'] == iv, 'Item_Weight'] = d_item_weight[iv]
            item_id_values = TestData.ix[TestData['Item_Weight'].isnull() == True,'Item_Identifier'].value_counts().index.values
            for iv in item_id_values:
                TestData.loc[TestData['Item_Identifier'] == iv, 'Item_Weight'] = d_item_weight[iv]
            ##
            d_flat_values = {}
            d_values = {}
            d_id_mean_values = {}
            d_id_median_values = {}
            for col in cls._l_cate_cols:
                d_values[col] = {}
                ##
                ValueCounts = TrainData[col].value_counts()
                NullValueCounts = TrainData[col].isnull().value_counts()
                if(True in NullValueCounts.index.values):
                    NullCount = NullValueCounts[True]
                else:
                    NullCount = 0
                vals = ValueCounts.index.values
                ##
                for v in vals:
                    d_flat_values['%s:%s' % (col, v)] = ValueCounts[v]
                if(NullCount > 0):
                    d_flat_values['%s:%s' % (col, 'missing')] = NullCount
                ##
                for i in range(len(vals)):
                    d_values[col][vals[i]] = i
                ##
                if(col in ['Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type']):
                    if(col not in d_id_mean_values):
                        d_id_mean_values[col] = {}
                        d_id_median_values[col] = {}
                    for v in vals:
                        d_id_mean_values[col][v] = np.mean(TrainData.ix[TrainData[col] == v, 'Item_Outlet_Sales'])
                        d_id_median_values[col][v] = np.median(TrainData.ix[TrainData[col] == v, 'Item_Outlet_Sales'])
            for id in d_id_mean_values:
                for val in d_id_mean_values[id]:
                    TrainData.loc[TrainData[id] == val, '%s:mean' % id] = (d_id_mean_values[id][val])
                    TestData.loc[TestData[id] == val, '%s:mean' % id] = (d_id_mean_values[id][val])
            for id in d_id_median_values:
                for val in d_id_median_values[id]:
                    TrainData.loc[TrainData[id] == val, '%s:median' % id] = (d_id_median_values[id][val])
                    TestData.loc[TestData[id] == val, '%s:median' % id] = (d_id_median_values[id][val])
            TrainData['Item_MRP'] = (TrainData['Item_MRP'])
            TestData['Item_MRP'] = (TestData['Item_MRP'])
            with open('%s/category.pkl' % OutputDir, 'wb') as ca_file,\
                open('%s/featmap.pkl' % OutputDir, 'wb') as fe_file,\
                open('%s/idmean.pkl' % OutputDir, 'wb') as im_file, \
                open('%s/idmedian.pkl' % OutputDir, 'wb') as im2_file:
                pickle.dump(d_values, ca_file, -1)
                pickle.dump(d_flat_values, fe_file, -1)
                pickle.dump(d_id_mean_values, im_file, -1)
                pickle.dump(d_id_median_values, im2_file, -1)
            ca_file.close()
            fe_file.close()
            im_file.close()
            im2_file.close()
            ## random split
            np.random.seed(2017)

            msk = np.random.rand(len(TrainData)) < cls._HoldoutRatio
            if('index' in TrainData.columns):
                TrainData.drop('index', axis= 1, inplace= True)
            if('index' in TestData.columns):
                TestData.drop('index', axis= 1, inplace= True)
            TrainData.reset_index(inplace = True)
            TestData.reset_index(inplace = True)
            ##
            HoldoutData = TrainData[msk].reset_index(drop= True)
            TrainValidData = TrainData[~msk].reset_index(drop= True)
            print('Train size %s, test size %s' % (len(TrainValidData), len(HoldoutData)))
            ## save
            with open('%s/test.pkl' % OutputDir, 'wb') as te_file,\
                open('%s/train.pkl' % OutputDir, 'wb') as tr_file,\
                open('%s/train_t.pkl' % OutputDir,'wb') as tr_file_t,\
                open('%s/test_t.pkl' % OutputDir, 'wb') as te_file_t:
                pickle.dump(TrainValidData, tr_file, -1)
                pickle.dump(HoldoutData, te_file, -1)
                pickle.dump(TrainData, tr_file_t, -1)
                pickle.dump(TestData, te_file_t, -1)
            tr_file.close()
            te_file.close()
            tr_file_t.close()
            te_file_t.close()
        elif(task == 'kfold'):
            """"""
            with open('%s/train_t.pkl' % InputDir, 'rb') as tr_file, \
                open('%s/test_t.pkl' % InputDir, 'rb') as te_file:
                TrainData_t = pickle.load(tr_file)
                TestData_t = pickle.load(te_file)
            tr_file.close()
            te_file.close()
            with open('%s/train.pkl' % OutputDir, 'wb') as tr_file, \
                open('%s/test.pkl' % OutputDir, 'wb') as te_file:
                pickle.dump(TrainData_t, tr_file, -1)
                pickle.dump(TestData_t, te_file, -1)
            tr_file.close()
            te_file.close()

            with open('%s/train.pkl' % InputDir, 'rb') as i_file:
                TrainData = pickle.load(i_file)
            i_file.close()
            if('index' in TrainData.columns):
                TrainData.drop('index', axis= 1, inplace= True)
            TrainData.reset_index(inplace = True)
            print('%s' % len(TrainData))
            for fold in range(cls._kfold):
                ## split
                VaData = TrainData[TrainData['index'] % cls._kfold == fold].reset_index(drop= True)
                TrData = TrainData[TrainData['index'] % cls._kfold != fold].reset_index(drop= True)
                print('%s => %s : %s : %s' % (fold, len(TrData), len(VaData), len(TrData) + len(VaData)))
                FoldOutputDir = '%s/%s' % (OutputDir, fold)
                ## save
                if (os.path.exists(FoldOutputDir) == False):
                    os.makedirs(FoldOutputDir)
                with open('%s/train.pkl' % FoldOutputDir, 'wb') as tr_file, open('%s/test.pkl' % FoldOutputDir, 'wb') as va_file:
                    pickle.dump(TrData, tr_file, -1)
                    pickle.dump(VaData, va_file, -1)
                tr_file.close()
                va_file.close()


        end = time.time()
        print('----- Task %s done, time consumed %ds' % (task, (end - start)))

        return

    @classmethod
    def run(cls, tasks, Input, Output):
        """"""
        start = time.time()
        if (os.path.exists(Output) == False):
            os.makedirs(Output)

        for i in range(len(tasks)):
            task = tasks[i]
            if(i == 0):
                InputDir = Input
            else:
                InputDir = '%s/%s' % (Output, tasks[i - 1])
            OutputDir = '%s/%s' % (Output, tasks[i])

            cls.__LaunchTask(task, InputDir, OutputDir)

        end = time.time()
        print('\nAll tasks done, time consumed %ds' % (end - start))
