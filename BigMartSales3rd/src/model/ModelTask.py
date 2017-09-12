import time,os,json,gc

import numpy as np
import pandas as pd

from util.DataUtil import DataUtil

from model.LightGBM import LGB
from model.ElasticNet import EN
from model.LinearRegression import LR

class ModelTask:
    """"""
    _kfold = 5
    _d_strategy = {
        'lgb': LGB,
        'en': EN,
        'lr': LR
    }

    def __init__(self, level, strategies, d_params, input, output, data_format= 'pkl'):
        """"""
        self._level = level
        self._strategies = strategies
        self._d_params = d_params
        self._input = input
        self._output = output
        self._data_format = data_format
        if(os.path.exists(self._output) == False):
            os.makedirs(self._output)

    """
    Train all models of specified strategy.
    """
    def __LaunchTrainTask(self, strategy, d_params, InputDir, OutputDir):
        """"""
        ## copy holdout data file
        HoldoutFile = '%s/holdout/test.%s' % (InputDir, self._data_format)
        HoldoutOutputDir = '%s/holdout' % OutputDir
        if (os.path.exists(HoldoutOutputDir) == False):
            os.makedirs(HoldoutOutputDir)
        DataUtil.save(DataUtil.load(HoldoutFile, self._data_format), '%s/test.%s' % (HoldoutOutputDir, self._data_format),format= self._data_format)
        print('\n ---- Copying holdout data done.\n')
        ## copy submit data file
        SubmitFile = '%s/kfold/test.%s' % (InputDir, self._data_format)
        SubmitOutputDir = '%s/submit' % OutputDir
        if (os.path.exists(SubmitOutputDir) == False):
            os.makedirs(SubmitOutputDir)
        DataUtil.save(DataUtil.load(SubmitFile, self._data_format),
                      '%s/test.%s' % (SubmitOutputDir, self._data_format), format=self._data_format)
        print('\n ---- Copying submit data done.\n')

        ## run each group of parameters of each algorithm
        OutputParams = []
        for variant in d_params['variants']:
            #### for check
            # if(variant['objective'] not in ['regression_l2', 'regression_l1']):
            #    continue
            # if(variant['selection'] not in ['random']):
            #    continue
            ####
            l_var_evals = []
            BestSTD = -1
            BestParam = {}
            BestRMSE = 65535
            VariantParams = []
            for param in d_params['params']:
                count = 0
                for VarKey in variant:
                    if ((VarKey in param) and (variant[VarKey] == param[VarKey])):
                        count += 1
                if (count == len(variant)):
                    VariantParams.append(param)
            print('\n==== Model tuning for strategy %s, variant %s begins... ====\n' % (strategy, variant))
            for param in VariantParams:
                model = self._d_strategy[strategy](param, self._kfold, InputDir, OutputDir, self._data_format)
                rmse = model.train()
                l_var_evals.append({'params': param, 'eval': rmse})
                del model
                gc.collect()
                mean = np.mean(list(rmse.values()))
                std = np.std(list(rmse.values()))
                if (mean < BestRMSE):
                    BestRMSE = mean
                    BestParam = param
                    BestSTD = std
            print('\n==== Model tuning for strategy %s, variant %s ends. ====\n' % (strategy, variant))
            OutputParams.append({'variant': variant, 'result': l_var_evals,'best': {'mean': BestRMSE, 'std': BestSTD, 'params': BestParam}})

        return OutputParams

    """
    Inferring for all models of specified strategy.
    """
    def __LaunchInferTask(self, strategy, InputDir, OutputDir):
        ## load best parameters
        with open('%s/params.txt' % InputDir, 'r') as i_file:
            params = json.load(i_file)
        i_file.close()
        print('\n---- Load parameters done. ----\n')

        ## load holdout test data
        HoldoutData = DataUtil.load('%s/holdout/test.%s' % (InputDir, self._data_format), format= self._data_format)
        print('\n---- Load holdout data done. ----\n')
        ## load submit test data
        SubmitData = DataUtil.load('%s/submit/test.%s' % (InputDir, self._data_format), format= self._data_format)
        print('\n---- Load submit data done. ----\n')

        variants = params[strategy]
        for variant in variants:
            print('\n==== Model inferring for strategy %s, variant %s begins. ====\n' % (strategy, variant['variant']))
            VarFields = variant['variant']
            varstr = ':'.join(['%s#%s' % (VarKey, VarFields[VarKey]) for VarKey in VarFields])
            best = variant['best']
            score = best['mean']
            BestParams = best['params']
            print('\n---- Best parameter ---- \n')
            print(score, BestParams)
            print('\n-------- n')

            head = 'strategy#%s:%s' % (strategy, varstr)
            VarOutputDir = '%s/%s' % (OutputDir, head)
            if (os.path.exists(VarOutputDir) == False):
                os.makedirs(VarOutputDir)
            model = self._d_strategy[strategy](BestParams, self._kfold, InputDir, VarOutputDir, self._data_format)
            if(self._level > 1):
                model.infer(head, HoldoutData, SubmitData, True)
            else:
                model.infer(head, HoldoutData, SubmitData, False)
            print('\n==== Model inferring for strategy %s, variant %s ends. ====\n' % (strategy, variant['variant']))
            del model
            gc.collect()

        return

    """ 
    Agggregate all models of strategies in current layer.
    """
    def __LaunchAggregateTask(self, strategies, InputDir, OutputDir):
        """"""
        l_variant_model = []
        for strategy in strategies:
            for variant in self._d_params['L%s' % self._level][strategy]['variants']:
                variant_model = ':'.join(['%s#%s' % (VarKey, variant[VarKey]) for VarKey in variant])
                l_variant_model.append('strategy#%s:%s' % (strategy, variant_model))
        #### for folds data
        for fold in range(self._kfold):
            print('\n Aggregate for fold %s begins. \n' % fold)
            l_train_fold = []
            l_test_fold = []
            ## load
            for mf in l_variant_model:
                TrainFile = '%s/%s/kfold/%s/train.%s' % (InputDir, mf, fold, self._data_format)
                TestFile = '%s/%s/kfold/%s/test.%s' % (InputDir, mf, fold, self._data_format)
                TrainData = DataUtil.load(TrainFile, self._data_format)
                TestData = DataUtil.load(TestFile, self._data_format)
                l_train_fold.append(TrainData)
                l_test_fold.append(TestData)
            print('\n Load data for fold %s done. \n' % fold)
            ## aggregate for train
            TrainFoldData = pd.DataFrame(index=l_train_fold[0].index)
            TrainFoldData['index'] = l_train_fold[0]['index']
            TrainFoldData['Item_Outlet_Sales'] = l_train_fold[0]['Item_Outlet_Sales']
            for idx in range(len(l_variant_model)):
                TrainFoldData[l_variant_model[idx]] = l_train_fold[idx][l_variant_model[idx]]
            #print('\n Aggregate for train done. \n')
            ## aggregate for test
            TestFoldData = pd.DataFrame(index=l_test_fold[0].index)
            TestFoldData['index'] = l_test_fold[0]['index']
            TestFoldData['Item_Outlet_Sales'] = l_test_fold[0]['Item_Outlet_Sales']
            for idx in range(len(l_variant_model)):
                TestFoldData[l_variant_model[idx]] = l_test_fold[idx][l_variant_model[idx]]
            #print('\n Aggregate for test done. \n')
            ## save
            FoldOutputDir = '%s/kfold/%s' % (OutputDir, fold)
            if (os.path.exists(FoldOutputDir) == False):
                os.makedirs(FoldOutputDir)
            DataUtil.save(TrainFoldData, '%s/train.%s' % (FoldOutputDir, self._data_format), format='csv')
            DataUtil.save(TestFoldData, '%s/test.%s' % (FoldOutputDir, self._data_format), format='csv')
            #print('\n Save aggregated fold %s data done. \n' % fold)
            print('\n Aggregate or fold %s done. \n' % fold)
        print('\n Aggregate kfold data fone.\n')
        ##### aggregate for holdout
        l_holdout = []
        for mf in l_variant_model:
            HoldoutFile = '%s/%s/holdout/test.%s' % (InputDir, mf, self._data_format)
            holdout = DataUtil.load(HoldoutFile, self._data_format)
            l_holdout.append(holdout)
        HoldoutData = pd.DataFrame(index=l_holdout[0].index)
        HoldoutData['index'] = l_holdout[0]['index']
        HoldoutData['Item_Outlet_Sales'] = l_holdout[0]['Item_Outlet_Sales']
        for idx in range(len(l_variant_model)):
            HoldoutData[l_variant_model[idx]] = l_holdout[idx][l_variant_model[idx]]
        ## save
        HoldoutOutputDir = '%s/holdout' % OutputDir
        if (os.path.exists(HoldoutOutputDir) == False):
            os.makedirs(HoldoutOutputDir)
        DataUtil.save(HoldoutData, '%s/test.%s' % (HoldoutOutputDir, self._data_format), format='csv')
        print('\n Aggregate for holdout data done.\n')
        #### aggregate for submit data
        l_submit = []
        for mf in l_variant_model:
            SubmitFile = '%s/%s/submit/test.%s' % (InputDir, mf, self._data_format)
            submit = DataUtil.load(SubmitFile, self._data_format)
            l_submit.append(submit)
        SubmitData = pd.DataFrame(index= l_submit[0].index)
        for idx in range(len(l_variant_model)):
            SubmitData[l_variant_model[idx]] = l_submit[idx][l_variant_model[idx]]
        ## save
        SubmitOutputDir = '%s/submit' % OutputDir
        if (os.path.exists(SubmitOutputDir) == False):
            os.makedirs(SubmitOutputDir)
        DataUtil.save(SubmitData, '%s/test.%s' % (SubmitOutputDir, self._data_format), format='csv')
        print('\n Aggregate for submit data done.\n')

        return

    def run(self):
        """"""
        d_level_params = self._d_params['L%s' % self._level]
        tasks = ['train', 'infer', 'aggregate']
        ss = time.time()
        for i in range(len(tasks)):
            ##
            if (i == 0):
                InputDir = self._input ## first task
            else:
                InputDir = '%s/%s' % (self._output, tasks[i - 1]) ## non-first task, new input is output directory of the last task
            OutputDir = '%s/%s' % (self._output, tasks[i])
            if (os.path.exists(OutputDir) == False):
                os.makedirs(OutputDir)

            s = time.time()
            ## deal task
            if(i < len(tasks) - 1):
                d_strategy_result= {}
                for strategy in self._strategies:
                    d_strategy_params = d_level_params[strategy]
                    print('\n==== task %s for strategy %s begins ==== \n' % (tasks[i], strategy))
                    start = time.time()

                    if(tasks[i] == 'train'):
                        d_result = self.__LaunchTrainTask(strategy, d_strategy_params, InputDir, OutputDir)
                        d_strategy_result[strategy] = d_result
                    elif(tasks[i] == 'infer'):
                        self.__LaunchInferTask(strategy, InputDir, OutputDir)

                    end = time.time()
                    print('\n==== task %s for strategy %s ends ==== \n' % (tasks[i], strategy))
                ## backup for checking
                if(tasks[i] == 'train'): ## train, save training results and parameters
                    with open('%s/params.txt' % OutputDir, 'w') as o_file:
                        json.dump(d_strategy_result, o_file, ensure_ascii=True, indent=4)
                    o_file.close()
            else:
                self.__LaunchAggregateTask(self._strategies, InputDir, OutputDir)

            e = time.time()
            print('\n ==== Task for %s done, time consumed %s. ====\n' % (tasks[i], (e - s)))
        ee = time.time()
        print('\n ==== All tasks done, time consumed %s ====\n' % (ee - ss))
