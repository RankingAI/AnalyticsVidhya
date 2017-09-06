import time
import pandas as pd
import os,sys
import dill as pickle
from feat.MissingValue import  MissingValue

class FeatureEngineering:
    """"""
    _InputDir = ''
    _OutputDir = ''

    _kfold = 8

    ## composite classes
    _missing = MissingValue

    def __init__(self, InputDir, OutputDIr):
        """"""
        self._InputDir = InputDir
        self._OutputDir = OutputDIr

    def __LaunchTask(self, task, mode):
        """"""
        print('\n---- Task %s begins ...' % (task))
        start = time.time()

        if(task == 'FeatureEncoding'):
            """"""
        elif(task == 'MissingValue'):
            """"""
            if(mode == 'kfold'):
                self.TrainData = self._missing.impute(self.TrainData)
            self.TestData = self._missing.impute(self.TestData)
            ## check
            #print(self.TrainData['Item_Weight'].isnull().value_counts())
            #print(self.TrainData['Outlet_Size'].isnull().value_counts())

        end = time.time()

        print('---- Task %s done, time consumed %ds' % (task, (end - start)))

        return

    def run(self, tasks):
        """"""
        ## for kfold
        print('\n==== Engineering for kfold begins ====')

        KFoldInputDir = '%s/kfold' % self._InputDir
        KFoldOutputDir = '%s/kfold' % self._OutputDir
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
                self.__LaunchTask(task, 'kfold')

            ## save
            if (os.path.exists(FoldOutputDir) == False):
                os.makedirs(FoldOutputDir)
            with open('%s/train.pkl' % FoldOutputDir, 'wb') as tr_file, open('%s/test.pkl' % FoldOutputDir, 'wb') as te_file:
                pickle.dump(self.TrainData, tr_file, -1)
                pickle.dump(self.TestData, te_file, -1)
            tr_file.close()
            te_file.close()

            print('\n==== fold %s done.' % fold)

        print('\n==== Engineering for kfold done ====')

        #### for holdout, local test

        print('\n==== Engineering for holdout begins ====')

        ## load
        HoldoutInputDir = '%s/holdout' % self._InputDir
        HoldoutOutputDir = '%s/holdout' % self._OutputDir
        with open('%s/test.pkl' % HoldoutInputDir, 'rb') as te_file:
            self.TestData = pickle.load(te_file)
        te_file.close()

        ## launch task
        for task in tasks:
            self.__LaunchTask(task, 'holdout')

        ## save
        if (os.path.exists(HoldoutOutputDir) == False):
            os.makedirs(HoldoutOutputDir)
        with open('%s/test.pkl' % HoldoutOutputDir, 'wb') as te_file:
            pickle.dump(self.TestData, te_file, -1)
        te_file.close()

        print('\n==== Engineering for holdout done ====')

        return
