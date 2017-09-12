import time,os,json,gc

import numpy as np
import pandas as pd

from util.DataUtil import DataUtil

from model.LightGBM import LGB
from model.ElasticNet import EN
from model.LinearRegression import LR

class ModelTask:
    """"""
    kfold = 5

    @classmethod
    def __LaunchTask(cls, strategy, d_params, task, InputDir, OutputDir, data_format, metric_pk):
        d_strategy = {
            'lgb': LGB,
            'en': EN,
            'lr': LR
        }

        print('\n ==== Strategy %s, task %s begins. ==== \n' % (strategy, task))
        start = time.time()

        if(task == 'train'):
            ## copy holdout data file
            HoldoutFile = '%s/holdout/test.%s' % (InputDir, data_format)
            HoldoutOutputDir = '%s/holdout' % OutputDir
            if(os.path.exists(HoldoutOutputDir) == False):
                os.makedirs(HoldoutOutputDir)
            DataUtil.save(DataUtil.load(HoldoutFile, data_format),'%s/test.%s' % (HoldoutOutputDir, data_format), format= data_format)
            print('\n ---- Copying holdout data done.\n')

            ## run each group of parameters of each algorithm
            OutputParams = {}
            OutputParams[strategy] = []
            for variant in d_params['variants']:
                #### for check
                #if(variant['objective'] not in ['regression_l2', 'regression_l1']):
                #    continue
                #if(variant['selection'] not in ['random']):
                #    continue
                ####
                l_var_evals = []
                VariantParams = []
                BestSTD = -1
                BestParam = {}
                BestRMSE = 65535
                for param in d_params['params']:
                    count = 0
                    for VarKey in variant:
                        if((VarKey in param) and (variant[VarKey] == param[VarKey])):
                            count += 1
                    if(count == len(variant)):
                        VariantParams.append(param)
                print('\n==== Model tuning for strategy %s, variant %s begins... ====\n' % (strategy, variant))
                for param in VariantParams:
                    model = d_strategy[strategy](param, cls.kfold, InputDir, OutputDir, data_format)
                    rmse = model.train()
                    l_var_evals.append({'params': param, 'eval': rmse})
                    del model
                    gc.collect()
                    mean = np.mean(list(rmse.values()))
                    std = np.std(list(rmse.values()))
                    if(mean < BestRMSE):
                        BestRMSE = mean
                        BestParam = param
                        BestSTD = std
                print('\n==== Model tuning for strategy %s, variant %s ends. ====\n' % (strategy, variant))
                OutputParams[strategy].append({'variant': variant, 'result': l_var_evals, 'best': {'mean': BestRMSE, 'std': BestSTD, 'params': BestParam}})
            with open('%s/params.txt' % (OutputDir),'w') as o_file:
                json.dump(OutputParams, o_file, ensure_ascii= True, indent= 4)
            o_file.close()
        elif(task == 'infer'):
            ## load best parameters
            with open('%s/params.txt' % InputDir, 'r') as i_file:
                params = json.load(i_file)
            i_file.close()
            print('\n---- Load parameters done. ----\n')

            ## load holdout test data
            HoldoutData = DataUtil.load('%s/holdout/test.%s' % (InputDir, data_format), format= 'csv')
            print('\n---- Load holdout data done. ----\n')

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
                if(os.path.exists(VarOutputDir) == False):
                    os.makedirs(VarOutputDir)
                model = d_strategy[strategy](BestParams, cls.kfold, InputDir, VarOutputDir, data_format)
                model.infer(head, HoldoutData, metric_pk)
                print('\n==== Model inferring for strategy %s, variant %s ends. ====\n' % (strategy, variant['variant']))
                del model
                gc.collect()
        elif(task == 'aggregate'):
            """"""
            #ModelFiles = os.listdir(InputDir)
            l_variant_model = []
            for variant in d_params['variants']:
                variant_model = ':'.join(['%s#%s' % (VarKey, variant[VarKey]) for VarKey in variant])
                l_variant_model.append('strategy#%s:%s' % (strategy, variant_model))
            ## for folds data
            for fold in range(cls.kfold):
                l_train_fold = []
                l_test_fold = []
                ## load
                for mf in l_variant_model:
                    TrainFile = '%s/%s/kfold/%s/train.%s' % (InputDir, mf, fold, data_format)
                    TestFile = '%s/%s/kfold/%s/test.%s' % (InputDir, mf, fold, data_format)
                    TrainData = DataUtil.load(TrainFile, data_format)
                    TestData = DataUtil.load(TestFile, data_format)
                    l_train_fold.append(TrainData)
                    l_test_fold.append(TestData)
                ## aggregate for train
                TrainFoldData = pd.DataFrame(index= l_train_fold[0].index)
                TrainFoldData['index'] = l_train_fold[0]['index']
                TrainFoldData['Item_Outlet_Sales'] = l_train_fold[0]['Item_Outlet_Sales']
                for idx in range(len(l_variant_model)):
                    TrainFoldData[l_variant_model[idx]] = l_train_fold[idx][l_variant_model[idx]]
                ## aggregate for test
                TestFoldData = pd.DataFrame(index= l_test_fold[0].index)
                TestFoldData['index'] = l_test_fold[0]['index']
                TestFoldData['Item_Outlet_Sales'] = l_test_fold[0]['Item_Outlet_Sales']
                for idx in range(len(l_variant_model)):
                    TestFoldData[l_variant_model[idx]] = l_test_fold[idx][l_variant_model[idx]]
                ## save
                FoldOutputDir = '%s/kfold/%s' % (OutputDir, fold)
                if(os.path.exists(FoldOutputDir) == False):
                    os.makedirs(FoldOutputDir)
                DataUtil.save(TrainFoldData, '%s/train.%s' % (FoldOutputDir, data_format), format= 'csv')
                DataUtil.save(TestFoldData, '%s/test.%s' % (FoldOutputDir, data_format), format= 'csv')
            ## aggregate for holdout
            l_holdout = []
            for mf in l_variant_model:
                HoldoutFile = '%s/%s/holdout/test.%s' % (InputDir, mf, data_format)
                holdout = DataUtil.load(HoldoutFile, data_format)
                l_holdout.append(holdout)
            HoldoutData = pd.DataFrame(index= l_holdout[0].index)
            HoldoutData['index'] = l_holdout[0]['index']
            HoldoutData['Item_Outlet_Sales'] = l_holdout[0]['Item_Outlet_Sales']
            for idx in range(len(l_variant_model)):
                HoldoutData[l_variant_model[idx]] = l_holdout[idx][l_variant_model[idx]]
            ## save
            HoldoutOutputDir = '%s/holdout' % OutputDir
            if (os.path.exists(HoldoutOutputDir) == False):
                os.makedirs(HoldoutOutputDir)
            DataUtil.save(HoldoutData, '%s/test.%s' % (HoldoutOutputDir, data_format), format= 'csv')

        end = time.time()
        print('\n ==== Strategy %s, task %s ends, time consumed %ds ==== \n' % (strategy, task, (end - start)))

        return

    @classmethod
    def run(cls, strategy, d_params, input, output, data_format= 'pkl', metric_pk= False):
        """"""
        tasks = ['train', 'infer', 'aggregate']
        #tasks = ['train', 'infer']

        for i in range(len(tasks)):
            if (i == 0):
                InputDir = input
            else:
                InputDir = '%s/%s' % (output, tasks[i - 1])
            OutputDir = '%s/%s' % (output, tasks[i])
            if (os.path.exists(OutputDir) == False):
                os.makedirs(OutputDir)

            cls.__LaunchTask(strategy, d_params, tasks[i], InputDir, OutputDir, data_format, metric_pk)

        return
