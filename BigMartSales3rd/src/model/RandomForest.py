"""
BigMartSales3rd is a toy project for methodology validation of StackNet introduced by Marios Michailidis,
 more details you can check https://github.com/kaz-Anova/StackNet. This is also a light weighted StackNet framed with 2-level stack.
Created by yuanpingzhou on 09/13/2017, more discussions you can e-mail me through pingzhou.yuan1@gmail.com.math
"""

import time
import numpy as np
import pandas as pd
import os
import math

from model.ModelBase import ModelBase
from util.DataUtil import DataUtil
from sklearn.ensemble import RandomForestRegressor

class RF(ModelBase):
    """"""
    _l_drop_cols = ['Item_Outlet_Sales', 'index']

    ## training, parameter tuning for single L1
    def train(self, importance=False):
        """"""
        print('\n Training for parameters %s \n' % self.parameters)
        d_fold_val = {}
        for fold in range(self.kfold):
            print('Fold %s begins.' % fold)

            ## load data
            TrainFile = '%s/kfold/%s/train.%s' % (self.InputDir, fold, self.data_format)
            TestFile = '%s/kfold/%s/test.%s' % (self.InputDir, fold, self.data_format)
            self.TrainData = DataUtil.load(TrainFile, format=self.data_format)
            self.TestData = DataUtil.load(TestFile, format=self.data_format)

            ## train and predict on valid
            self.__fit()
            eval = self.__predict()
            d_fold_val[fold] = eval

            ## save
            OutputDir = '%s/kfold/%s' % (self.OutputDir, fold)
            if (os.path.exists(OutputDir) == False):
                os.makedirs(OutputDir)
            DataUtil.save(self.TrainData, '%s/train.%s' % (OutputDir, self.data_format), format=self.data_format)
            DataUtil.save(self.TestData, '%s/test.%s' % (OutputDir, self.data_format), format=self.data_format)

            print('Fold %d done. ' % fold)

        print('\n Training done.\n')

        return d_fold_val

    ## inferring for fold data and holdout data
    def infer(self, head, HoldoutData, SubmitData, metric_pk= False):
        """"""
        print('\n Inferring for %s begins. \n' % head)
        ##
        l_pred_fold = []
        PredHoldout = pd.DataFrame(index= HoldoutData.index)
        PredHoldout['index'] = HoldoutData['index']
        PredHoldout['Item_Outlet_Sales'] = HoldoutData['Item_Outlet_Sales']
        PredSubmit = pd.DataFrame(index= SubmitData.index)
        for fold in range(self.kfold):
            ## load
            TrainFile = '%s/kfold/%s/train.%s' % (self.InputDir, fold, self.data_format)
            TestFile = '%s/kfold/%s/test.%s' % (self.InputDir, fold, self.data_format)
            self.TrainData = DataUtil.load(TrainFile, format= self.data_format)
            self.TestData = DataUtil.load(TestFile, format= self.data_format)

            ## fit
            PredFold = pd.DataFrame(index= self.TestData.index)
            PredFold['index'] = self.TestData['index']
            PredFold['Item_Outlet_Sales'] = self.TestData['Item_Outlet_Sales']
            PredFold['fold'] = fold
            self.__fit()

            ## inferring
            PredFold[head] = self._model.predict(self.TestData[self._l_train_columns])
            PredHoldout['fold%s' % (fold)] = self._model.predict(HoldoutData[self._l_train_columns])
            PredSubmit['fold%s' % fold] = self._model.predict(SubmitData[self._l_train_columns])
            l_pred_fold.append(PredFold)
        ## aggregate folds data
        PredKFold = pd.concat(l_pred_fold, axis= 0, ignore_index= True)
        ## save for folds data
        for fold in range(self.kfold):
            FoldOutputDir = '%s/kfold/%s' % (self.OutputDir, fold)
            if(os.path.exists(FoldOutputDir) == False):
                os.makedirs(FoldOutputDir)
            TrainFile = '%s/train.%s' % (FoldOutputDir, self.data_format)
            TestFile = '%s/test.%s' % (FoldOutputDir, self.data_format)

            TrainData = PredKFold[PredKFold['fold'] != fold]
            TestData = PredKFold[PredKFold['fold'] == fold]
            DataUtil.save(TrainData, TrainFile, format= self.data_format)
            DataUtil.save(TestData, TestFile, format= self.data_format)

        HoldCols = [col for col in PredHoldout.columns if col.startswith('fold')]
        ## save for holdout data
        PredHoldout[head] = PredHoldout[HoldCols].mean(axis= 1)
        HoldoutOutputDir = '%s/holdout' % self.OutputDir
        if (os.path.exists(HoldoutOutputDir) == False):
            os.makedirs(HoldoutOutputDir)
        DataUtil.save(PredHoldout, '%s/test.%s' % (HoldoutOutputDir, self.data_format), format= self.data_format)
        ## save for submit data
        PredSubmit[head] = PredSubmit[HoldCols].mean(axis=1)
        SubmitOutputDir = '%s/submit' % self.OutputDir
        if (os.path.exists(SubmitOutputDir) == False):
            os.makedirs(SubmitOutputDir)
        DataUtil.save(PredSubmit, '%s/test.%s' % (SubmitOutputDir, self.data_format), format=self.data_format)

        ## metric PK
        if (metric_pk):
            d_metric = {}
            for col in self._l_train_columns:
                diff = (HoldoutData[col] - HoldoutData['Item_Outlet_Sales'])
                rmse = np.sqrt(np.sum(diff * diff) / len(diff))
                d_metric[col] = rmse
            diff = PredHoldout[head] - PredHoldout['Item_Outlet_Sales']
            ensemble_metric = np.sqrt(np.sum(diff * diff) / len(diff))
            print('\n===== metric pk result ====\n')
            print('single model: %s, ensemble model %s: %s' % (d_metric, head, ensemble_metric))
            print('\n===== metric pk result ====\n')

        print('\n Inferring don. \n')
        return

    ## L1 fitting
    def __fit(self):
        """"""
        start = time.time()
        ##
        id_cols = [col for col in self.TrainData.columns if (col.startswith('Item_Identifier'))]
        self._l_drop_cols.extend(id_cols)
        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['Item_Outlet_Sales']
        ##
        self._l_train_columns = X.columns
        print('Size of feature space: %s' % len(self._l_train_columns))
        ##
        self._model = RandomForestRegressor(
            criterion= self.parameters['criterion'],
            n_estimators= self.parameters['n_estimators'],
            max_depth= self.parameters['max_depth'],
            max_features= self.parameters['max_features'],
        )
        self._model.fit(X, Y)
        ##
        end = time.time()
        print('Fitting done. Time elapsed %ds' % (end - start))

        return

    ## predict
    def __predict(self):
        """"""
        start = time.time()
        ##
        x_test = self.TestData[self._l_train_columns]
        pred_test = self._model.predict(x_test)
        truth_test = self.TestData['Item_Outlet_Sales']
        ## RMSE
        diff = (pred_test - truth_test)
        rmse = np.sqrt(np.sum(diff * diff) / len(diff))

        ##
        end = time.time()
        print('Prediction done. Time consumed %ds' % (end - start))

        return rmse
