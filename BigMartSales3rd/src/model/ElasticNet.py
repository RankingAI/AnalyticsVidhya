import gc
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from model import ModelBase


class EN(ModelBase):

   _l_drop_cols = ['logerror', 'parcelid', 'transactiondate', 'index','nullcount']
   #_l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index','nullcount', 'taxdelinquencyyear', 'finishedsquarefeet15', 'finishedsquarefeet6', 'yardbuildingsqft17']
   _alpha = 0.001
   _ratio = 0.1
   _iter = 1000
   _sel = 'random'

   """"""
   def __fit(self, params = {}):
      """"""
      start = time.time()
      ##
      id_cols = [col for col in self.TrainData.columns if (col.startswith('Item_Identifier'))]
      self._l_drop_cols.extend(id_cols)

      X = self.TrainData.drop(self._l_drop_cols, axis=1)
      Y = self.TrainData['Item_Outlet_Sales']
      self._l_train_columns = X.columns
      X = X.values.astype(np.float32, copy=False)

      en = ElasticNet(alpha= self._alpha, l1_ratio= self._ratio, max_iter= self._iter, tol= 1e-4, selection= self._sel, random_state= 2017)
      self._model = en.fit(X, Y)

      end = time.time()
      print('train done, time consumed %ds' % (end - start))

      return

   def evaluate(self):
      """"""
      ValidData = self.ValidData

      #ValidData['bathroomratio'] = ValidData['bathroomcnt'] / ValidData['calculatedbathnbr']
      #ValidData.loc[ValidData['bathroomratio'] < 0, 'bathroomratio'] = -1
      ValidData['longitude'] -= -118600000
      ValidData['latitude'] -= 34220000

      # ValidData['structuretaxvalueratio'] = ValidData['structuretaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
      # ValidData['landtaxvalueratio'] = ValidData['landtaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
      # ValidData.loc[ValidData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
      # ValidData.loc[ValidData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

      ## not truncate outliers
      pred_valid = pd.DataFrame(index= ValidData.index)
      pred_valid['parcelid'] = ValidData['parcelid']

      truth_valid = pd.DataFrame(index= ValidData.index)
      truth_valid['parcelid'] = ValidData['parcelid']

      start = time.time()

      for d in self._l_valid_predict_columns:
         l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                            self._l_train_columns]
         x_valid = ValidData[l_valid_columns]
         x_valid = x_valid.values.astype(np.float32, copy=False)
         pred_valid[d] = self._model.predict(x_valid)# * 0.80 + 0.011 * 0.20
         df_tmp = ValidData[ValidData['transactiondate'].dt.month == int(d[-2:])]
         truth_valid.loc[df_tmp.index, d] = df_tmp['logerror']

      score = 0.0
      ae = np.abs(pred_valid - truth_valid)
      for col in ae.columns:
         score += np.sum(ae[col])
      score /= len(pred_valid)  ##!! divided by number of instances, not the number of 'cells'
      print('============================= ')
      print('Local MAE is %.6f' % score)
      print('=============================')

      end = time.time()

      del self.ValidData
      gc.collect()

      print('time elapsed %ds' % (end - start))

   def submit(self):
      """"""
      ## retrain with the whole training data
      self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]

      self.TrainData['longitude'] -= -118600000
      self.TrainData['latitude'] -= 34220000

      X = self.TrainData.drop(self._l_drop_cols, axis=1)
      Y = self.TrainData['logerror']
      X = X.values.astype(np.float32, copy=False)

      en = ElasticNet(alpha= self._alpha, l1_ratio = self._ratio, max_iter= self._iter, tol= 1e-4, selection= self._sel, random_state= 2017)
      self._model = en.fit(X, Y)

      del self.TrainData, X, Y
      gc.collect()

      self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')
      #self.TestData = self.TestData.sample(frac = 0.01)

      self._sub = pd.DataFrame(index=self.TestData.index)
      self._sub['ParcelId'] = self.TestData['parcelid']

      self.TestData['longitude'] -= -118600000
      self.TestData['latitude'] -= 34220000
      N = 200000
      start = time.time()
      for d in self._l_test_predict_columns:
         s0 = time.time()

         print('Prediction for column %s ' % d)
         l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                           self._l_train_columns]
         x_test = self.TestData[l_test_columns]

         for idx in range(0, len(x_test), N):
            x_test_block = x_test[idx:idx + N].values.astype(np.float32, copy=False)
            ret = self._model.predict(x_test_block)# * 0.99 + 0.011 * 0.01
            self._sub.loc[x_test[idx:idx + N].index, d] = ret
            print(np.mean(np.abs(ret)))

         e0 = time.time()
         print('Prediction for column %s is done. time elapsed %ds' % (d, (e0 - s0)))

      ## clean
      del self.TestData
      gc.collect()

      end = time.time()
      print('Prediction is done. time elapsed %ds' % (end - start))

      if (os.path.exists(self.OutputDir) == False):
         os.makedirs(self.OutputDir)

      self._sub.to_csv(
         '{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__, datetime.now().strftime('%Y%m%d-%H:%M:%S')),
         index=False, float_format='%.4f')