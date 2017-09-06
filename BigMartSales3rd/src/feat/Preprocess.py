import time
import numpy as np
import pandas as pd
import os,sys
import dill as pickle
from sklearn.cross_validation import KFold

class Preprocess:
    """"""
    _HoldoutRatio = 0.1
    _kfold = 8

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
            ## random split
            np.random.seed(2017)
            msk = np.random.rand(len(TrainData)) < cls._HoldoutRatio
            HoldoutData = TrainData[msk]
            TrainData = TrainData[~msk]
            print('Train size %s, test size %s' % (len(TrainData), len(HoldoutData)))
            ## save
            with open('%s/test.pkl' % OutputDir, 'wb') as te_file, open('%s/train.pkl' % OutputDir, 'wb') as tr_file:
                pickle.dump(TrainData, tr_file, -1)
                pickle.dump(HoldoutData, te_file, -1)
            tr_file.close()
            te_file.close()
        elif(task == 'kfold'):
            """"""
            with open('%s/train.pkl' % InputDir, 'rb') as i_file:
                TrainData = pickle.load(i_file)
            i_file.close()
            TrainData.reset_index(inplace = True)
            print('%s' % len(TrainData))
            # kf = KFold(len(TrainData), n_folds= cls._kfold, random_state= 2017, shuffle= True)
            # for fold, (TrainIndex, ValidIndex) in enumerate(kf, start= 1):
            #     TrData, VaData = TrainData[TrainData['index'].isin(TrainIndex)], TrainData[TrainData['index'].isin(ValidIndex)]
            for fold in range(cls._kfold):
                ## split
                VaData = TrainData[TrainData['index'] % cls._kfold == fold]
                TrData = TrainData[TrainData['index'] % cls._kfold != fold]
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
