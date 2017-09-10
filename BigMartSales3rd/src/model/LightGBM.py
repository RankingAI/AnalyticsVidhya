import time
import lightgbm
import numpy as np
import pandas as pd
import sys
import os
import dill as pickle
from datetime import datetime
from model.ModelBase import ModelBase

class LGB(ModelBase):
    """"""
    _max_bin = 31

    _l_drop_cols = ['Item_Outlet_Sales', 'index']

    ## training, parameter tuning for single model
    def train(self, importance = False):
        """"""
        print('\n parameters %s n' % self.parameters)
        d_fold_val = {}
        for fold in range(self.kfold):
            print('\n---- fold %s begins.\n' %fold)
            ## load data
            with open('%s/kfold/%s/train.pkl' % (self.InputDir, fold), 'rb') as tr_file, \
                    open('%s/kfold/%s/test.pkl' % (self.InputDir, fold), 'rb') as te_file:
                self.TrainData = pickle.load(tr_file)
                self.TestData = pickle.load(te_file)
            tr_file.close()
            te_file.close()
            ## train and predict on valid
            self.__fit()
            eval = self.__predict()

            d_fold_val[fold] = eval

            ## save
            OutputDir = '%s/kfold/%s' % (self.OutputDir, fold)
            if (os.path.exists(OutputDir) == False):
                os.makedirs(OutputDir)
            with open('%s/train.pkl' % OutputDir,'wb') as tr_file,\
                open('%s/test.pkl' % OutputDir, 'wb') as te_file:
                pickle.dump(self.TrainData, tr_file, -1)
                pickle.dump(self.TestData, te_file, -1)
            tr_file.close()
            te_file.close()
            print('\n---- Fold %d done. ----\n' % fold)

        return d_fold_val

    ## inferring for fold data and holdout data
    def infer(self, head, HoldoutData):
        """"""
        ##
        PredHoldout = pd.DataFrame(index= HoldoutData.index)
        PredHoldout['index'] = HoldoutData['index']
        PredHoldout['Item_Outlet_Sales'] = HoldoutData['Item_Outlet_Sales']
        for fold in range(self.kfold):
            ## load
            with open('%s/kfold/%s/train.pkl' % (self.InputDir, fold), 'rb') as tr_file, \
                    open('%s/kfold/%s/test.pkl' % (self.InputDir, fold), 'rb') as te_file:
                self.TrainData = pickle.load(tr_file)
                self.TestData = pickle.load(te_file)
            tr_file.close()
            te_file.close()
            ## fit
            PredTest = pd.DataFrame(index= self.TestData.index)
            PredTest['index'] = self.TestData['index']
            self.__fit()
            ## inferring for test data in folds
            x_test = self.TestData[self._l_train_columns]
            PredTest[head] = self._model.predict(x_test)
            ## inferring for holdout data
            PredHoldout['fold%s' % (fold)] = self._model.predict(HoldoutData[self._l_train_columns])
            ## save
            OutputDir = '%s/kfold' % (self.OutputDir)
            if (os.path.exists(OutputDir) == False):
                os.makedirs(OutputDir)
            PredTest.to_csv('%s/%s.csv' % (OutputDir, fold), float_format= '%.4f', index= False)
        HoldCols = [col for col in PredHoldout.columns if col.startswith('fold')]
        PredHoldout[head] = PredHoldout[HoldCols].mean(axis= 1)
        PredHoldout.to_csv('%s/holdout.csv' % (self.OutputDir), float_format='%.4f', index= False)

        return

    ## model fitting
    def __fit(self):
        """"""
        start = time.time()
        ##
        id_cols = [col for col in self.TrainData.columns if(col.startswith('Item_Identifier'))]
        self._l_drop_cols.extend(id_cols)
        X = self.TrainData.drop(self._l_drop_cols,axis= 1)
        Y = self.TrainData['Item_Outlet_Sales']
        ##
        self._l_train_columns = X.columns
        print('Size of feature space: %s' % len(self._l_train_columns))
        ##
        d_cv = lightgbm.Dataset(X.values, label=Y.values, max_bin= self._max_bin, silent= True, free_raw_data= True)
        self._model = lightgbm.train(self.parameters, d_cv)
        ##
        end = time.time()
        print('\nTraining is done. Time elapsed %ds' % (end - start))

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
        rmse = np.sqrt(np.sum(diff * diff)/len(diff))

        ##
        end = time.time()
        print('\n Prediction done. Time consumed %ds' % (end - start))

        return rmse
