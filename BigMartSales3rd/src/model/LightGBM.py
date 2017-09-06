import time
import lightgbm
import pandas as pd
import numpy as np
import dill as pickle
import os, sys
import gc
from datetime import datetime
from model.ModelBase import ModelBase

class LGB(ModelBase):
    """"""
    _params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'lambda_l1': 0.01,
        'sub_feature': 0.90,
        'bagging_fraction':  0.85,
        'num_leaves': 64,
        'min_data':  100,
        'min_hessian':  0.1,
        'learning_rate': 0.02,
        'bagging_freq': 15
    }

    _iter = 800

    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate']

    def train(self, importance = False):
        """"""
        # l_min_data = [100, 200]
        # l_num_leaves = [16]
        # MinMAE = 1000
        # BestMinData = -1
        # BestNumLeaves = -1
        # d_log = {}
        # for md in l_min_data:
        #     if(md not in d_log):
        #         d_log[md] = {}
        #     for nl in l_num_leaves:
        #         self._params['min_data'] = md
        #         self._params['num_leaves'] = nl
        #         self.__fit()
        #         mae = self.__predict('train')
        #         if(mae < MinMAE):
        #             MinMAE = mae
        #             BestMinData = md
        #             BestNumLeaves = nl
        #         d_log[md][nl] = mae
        #
        # print(d_log)
        # print('best min_data: %s, best num_leaves: %s' % (BestMinData, BestNumLeaves))
        #sys.exit(1)
        #
        # self._params['min_data'] = BestMinData
        # self._params['num_leaves'] = BestNumLeaves
        self.__fit()

        return self.__predict('train')

    def submit(self):
        """"""
        self.__fit()
        self.__predict('test')

        return

    def __fit(self):
        """"""
        start = time.time()

        ## truncate outliers
        print('initial data shape : ', self.TrainData.shape)
        TrainData = self.TrainData[(self.TrainData['logerror'] > self.MinLogError) & (self.TrainData['logerror'] < self.MaxLogError)]
        print('data shape truncated: ', TrainData.shape)

        ## ---- new features -----
        NewFeats = []
        ## lon/lat transformed
        TrainData['longitude'] -= -118600000
        TrainData['lat1'] = TrainData['latitude'] - 34500000
        TrainData['latitude'] -= 34220000

        #print(TrainData[['longitude', 'lat1', 'latitude']].head(50))

        ## add structure tax ratio
        TrainData['structuretaxvalueratio'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
        ## add land tax ratio
        TrainData['landtaxvalueratio'] = TrainData['landtaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        NewFeats.extend(['lat1', 'structuretaxvalueratio', 'landtaxvalueratio'])
        ##

        ##
        #TrainData['N-ValueRatio'] = TrainData['taxvaluedollarcnt'] / TrainData['taxamount']
        #TrainData['N-LivingAreaProp'] = TrainData['calculatedfinishedsquarefeet'] / TrainData['lotsizesquarefeet']
        #TrainData['N-ValueProp'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['landtaxvaluedollarcnt']
        #TrainData['N-TaxScore'] = TrainData['taxvaluedollarcnt'] * TrainData['taxamount']

        print('data shape after feature space being extended : ', TrainData.shape)
        if(len(self.l_selected_feats) > 0):
            n_size = len(NewFeats)
            NewFeats.extend(self.l_selected_feats)
            X = TrainData[NewFeats]
            print('Selected feature size %s, new feature size %s' % (len(self.l_selected_feats), n_size))
        else:
            X = TrainData.drop(self._l_drop_cols,axis= 1)
        Y = TrainData['logerror']

        self._l_train_columns = X.columns

        X = X.values.astype(np.float32, copy=False)
        d_cv = lightgbm.Dataset(X,label=Y, params= {'bin_construct_sample_cnt': 100000})

        self._model = lightgbm.train(self._params, d_cv, self._iter)
        print(d_cv.params)

        end = time.time()

        print('Training is done. Time elapsed %ds' % (end - start))
        #del self.TrainData
        #gc.collect()

        return

    def __predict(self, mode):
        """"""
        if(mode == 'train'):
            TestData = self.TestData.copy()
        else:
            del self.TestData
            gc.collect()
            print('Reload test data from hdf file...')
            TestData = pd.read_hdf(path_or_buf= '%s/test.hdf' % self.SubmitInputDir, key='test')
            print('Reloading done.')
        print('initial data shape: ', TestData.shape)

        ## ---- new features ----
        ## lon/lat transformed
        TestData['longitude'] -= -118600000
        TestData['lat1'] = TestData['latitude'] - 34500000
        TestData['latitude'] -= 34220000

        ## add structure tax ratio
        TestData['structuretaxvalueratio'] = TestData['structuretaxvaluedollarcnt'] / TestData['taxvaluedollarcnt']
        TestData.loc[TestData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
        ## add land tax ratio
        TestData['landtaxvalueratio'] = TestData['landtaxvaluedollarcnt'] / TestData['taxvaluedollarcnt']
        TestData.loc[TestData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        #print(self.TestData[['longitude', 'lat1', 'latitude']].head(50))
        ##

        ##
        #TestData['N-ValueRatio'] = TestData['taxvaluedollarcnt'] / TestData['taxamount']
        #TestData['N-LivingAreaProp'] = TestData['calculatedfinishedsquarefeet'] / TestData['lotsizesquarefeet']
        #TestData['N-ValueProp'] = TestData['structuretaxvaluedollarcnt'] / TestData['landtaxvaluedollarcnt']
        #TestData['N-TaxScore'] = TestData['taxvaluedollarcnt'] * TestData['taxamount']

        print('data shape after feature space being extended : ', TestData.shape)

        pred_test = pd.DataFrame(index = TestData.index)
        pred_test['parcelid'] = TestData['parcelid']

        if(mode == 'train'):
            truth_test = pd.DataFrame(index = TestData.index)
            truth_test['parcelid'] = TestData['parcelid']

        start = time.time()
        N = 200000
        for mth in self.l_test_predict_columns: ## objective columns need to be predicted
            s0 = time.time()

            FixedTestPredCols = ['%s%s' % (c, mth) if (c in ['lastgap']) else c for c in self._l_train_columns]
            x_test = TestData[FixedTestPredCols]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N].values.astype(np.float32, copy=False)
                #self._model.reset_parameter({"num_threads": 1})
                ret = self._model.predict(x_test_block)
                ## fill prediction
                pred_test.loc[x_test[idx:idx + N].index, '%s' % mth] = ret
                print('---------- %s' % idx)

            ## fill truth
            if(mode == 'train'):
                df_tmp = TestData[TestData['transactiondate'].dt.month == mth]
                truth_test.loc[df_tmp.index, '%s' % mth] = df_tmp['logerror']

            e0 = time.time()
            print('Prediction for column %s is done. time elapsed %ds' % (mth, (e0 - s0)))

        if(mode == 'train'):
            score = 0.0
            ae = np.abs(pred_test - truth_test)
            for col in ae.columns:
                score += np.sum(ae[col])
            score /= len(pred_test)  ##!! divided by number of instances, not the number of 'cells'
            print('============================= ')
            print('Local MAE is %.6f' % score)
            print('=============================')

            # ## save for predict
            # OutputFile = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
            # with open(OutputFile, 'wb') as o_file:
            #    pickle.dump(pred_test, o_file, -1)
            # o_file.close()
            # ## save for truth
            # with open('%s/truth.pkl' % self.OutputDir, 'wb') as o_file:
            #     pickle.dump(truth_test, o_file, -1)
            # o_file.close()

            end = time.time()
            print('time elapsed %ds' % (end - start))

            return score
        else:
            if (os.path.exists(self.OutputDir) == False):
                os.makedirs(self.OutputDir)

            pred_test.to_csv('{0}/{1}.csv'.format(self.OutputDir, self.__class__.__name__),index=False, float_format='%.4f')

            end = time.time()
            print('time elapsed %ds' % (end - start))

            return

