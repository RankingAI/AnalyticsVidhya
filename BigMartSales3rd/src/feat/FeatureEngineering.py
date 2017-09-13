import time
import pandas as pd
import os,sys
import dill as pickle
from feat.MissingValue import  MissingValue
from feat.FeatureEncoding import FeatureEncoding
from util.DataUtil import DataUtil

class FeatureEngineering:
    """"""
    _InputDir = ''
    _OutputDir = ''

    _kfold = 5

    ## composite classes
    _missing = MissingValue
    _encode = FeatureEncoding

    def __init__(self, InputDir, OutputDIr):
        """"""
        self._InputDir = InputDir
        self._OutputDir = OutputDIr

    def __LaunchTask(self, task, encode_type = 'simple'):
        """"""
        print('\n---- Task %s begins ...' % (task))
        start = time.time()

        if(task == 'FeatureEncoding'):
            """"""
            d_values = {}
            if(encode_type == 'simple'):
                d_values = self._d_values
            elif(encode_type == 'onehot'):
                d_values = self._d_flat_values
            elif(encode_type == 'mixed'):
                d_values = (self._d_flat_values, self._d_id_median_values)
            self.TrainData = self._encode.ordinal(self.TrainData, d_values, mode= encode_type)
            self.TestData = self._encode.ordinal(self.TestData, d_values, mode = encode_type)
            ## check
        elif(task == 'MissingValue'):
            """"""
            self.TrainData = self._missing.impute(self.TrainData)
            self.TestData = self._missing.impute(self.TestData)
            ## check
            #print(self.TrainData['Item_Weight'].isnull().value_counts())
            #print(self.TrainData['Outlet_Size'].isnull().value_counts())

        end = time.time()

        print('---- Task %s done, time consumed %ds' % (task, (end - start)))

        return

    def run(self, tasks, encode_type = 'simple'):
        """"""
        print('\n==== Engineering for kfold begins ====')
        ## load category values
        with open('%s/holdout/category.pkl' % self._InputDir, 'rb') as ca_file,\
            open('%s/holdout/featmap.pkl' % self._InputDir, 'rb') as fe_file,\
            open('%s/holdout/idmean.pkl' % self._InputDir, 'rb') as im_file,\
            open('%s/holdout/idmedian.pkl' % self._InputDir, 'rb') as im2_file:
            self._d_values = pickle.load(ca_file)
            self._d_flat_values = pickle.load(fe_file)
            self._d_id_mean_values = pickle.load(im_file)
            self._d_id_median_values = pickle.load(im2_file)
        ca_file.close()
        fe_file.close()
        im_file.close()
        im2_file.close()

        KFoldInputDir = '%s/kfold' % self._InputDir
        KFoldOutputDir = '%s/kfold' % self._OutputDir
        #### for submit, public test
        with open('%s/train.pkl' % KFoldInputDir, 'rb') as tr_file, open('%s/test.pkl' % KFoldInputDir, 'rb') as te_file:
            self.TrainData = pickle.load(tr_file)
            self.TestData = pickle.load(te_file)
        tr_file.close()
        te_file.close()
        for task in tasks:
            self.__LaunchTask(task, encode_type= encode_type)
        ## save submit, public test
        SubmitOutputDir = '%s/submit' % self._OutputDir
        if (os.path.exists(SubmitOutputDir) == False):
            os.makedirs(SubmitOutputDir)
        DataUtil.save(self.TrainData, '%s/train.csv' % SubmitOutputDir, format='csv')
        DataUtil.save(self.TestData, '%s/test.csv' % SubmitOutputDir, format='csv')
        #### for kfold, local CV
        for fold in range(self._kfold):
            print('\n==== fold %s begins ...' % fold)

            FoldInputDir = '%s/%s' % (KFoldInputDir, fold)
            FoldOutputDir = '%s/%s' % (KFoldOutputDir, fold)
            ## load
            with open('%s/train.pkl' % FoldInputDir, 'rb') as tr_file, open('%s/test.pkl' % FoldInputDir, 'rb') as te_file:
                self.TrainData = pickle.load(tr_file)
                self.TestData = pickle.load(te_file)
            tr_file.close()
            te_file.close()

            ## launch task
            for task in tasks:
                self.__LaunchTask(task, encode_type = encode_type)

            ## save
            if (os.path.exists(FoldOutputDir) == False):
                os.makedirs(FoldOutputDir)
            DataUtil.save(self.TrainData, '%s/train.csv' % FoldOutputDir, format='csv')
            DataUtil.save(self.TestData, '%s/test.csv' % FoldOutputDir, format='csv')

            print('\n==== fold %s done.' % fold)

        print('\n==== Engineering for kfold done ====')

        #### for holdout, local test
        print('\n==== Engineering for holdout begins ====')

        ## load
        HoldoutInputDir = '%s/holdout' % self._InputDir
        HoldoutOutputDir = '%s/holdout' % self._OutputDir
        with open('%s/test.pkl' % HoldoutInputDir, 'rb') as te_file,\
            open('%s/train.pkl' % HoldoutInputDir, 'rb') as tr_file:
            self.TestData = pickle.load(te_file)
            self.TrainData = pickle.load(tr_file)
        te_file.close()
        tr_file.close()

        ## launch task
        for task in tasks:
            self.__LaunchTask(task, encode_type = encode_type)

        ## save
        if (os.path.exists(HoldoutOutputDir) == False):
            os.makedirs(HoldoutOutputDir)
        DataUtil.save(self.TrainData, '%s/train.csv' % HoldoutOutputDir, format= 'csv')
        DataUtil.save(self.TestData, '%s/test.csv' % HoldoutOutputDir, format= 'csv')

        print('\n==== Engineering for holdout done ====')
        return
