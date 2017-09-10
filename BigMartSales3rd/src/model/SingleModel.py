from model.LightGBM import LGB
import sys
import json
import gc
import numpy as np
import os
import time
import pickle

class SingleModel:
    """"""
    kfold = 5

    @classmethod
    def __LaunchTask(cls, task, mode, InputDir, OutputDir, HoldoutFile):
        d_model = {
            'lgb': LGB
        }

        if(mode == 'train'):
            ## load parameters
            print('\n---- Loading parameters to be tuned. ----\n')
            with open('%s/parameters.txt' % InputDir, 'r') as i_file:
                d_params = eval(i_file.read())
            i_file.close()
            print('\n ---- Loading parameters done. ----\n')

            ## run each group of parameters of each algorithm
            OutputParams = {}
            OutputParams['L1'] = {}
            for algo in d_params['L1']:
                OutputParams['L1'][algo] = []
                ## for variant algorithms
                for variant in d_params['L1'][algo]['variants']:
                    #### for check
                    #if(variant['objective'] not in ['regression_l2', 'regression_l1']):
                    #    continue
                    ####
                    l_var_evals = []
                    VariantParams = []
                    BestSTD = -1
                    BestParam = {}
                    BestRMSE = 65535
                    for param in d_params['L1'][algo]['params']:
                        count = 0
                        for VarKey in variant:
                            if((VarKey in param) and (variant[VarKey] == param[VarKey])):
                                count += 1
                        if(count == len(variant)):
                            VariantParams.append(param)
                    print('\n==== Model tuning for level %s, algo %s, variant %s begins... ====\n' % ('L1', algo, variant))
                    for param in VariantParams:
                        model = d_model[task](param, cls.kfold, InputDir, OutputDir)
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
                    print('\n==== Model tuning for level %s, algo %s, variant %s ends. ====\n' % ('L1', algo, variant))
                    OutputParams['L1'][algo].append({'variant': variant, 'result': l_var_evals, 'best': {'mean': BestRMSE, 'std': BestSTD, 'params': BestParam}})
            with open('%s/params.txt' % (OutputDir),'w') as o_file:
                json.dump(OutputParams, o_file, ensure_ascii= True, indent= 4)
            o_file.close()
        elif(mode == 'infer'):
            ## load best parameters
            with open('%s/params.txt' % InputDir, 'r') as i_file:
                params = json.load(i_file)
            i_file.close()
            print('\n---- Load parameters done. ----\n')
            ## load holdout test data
            with open(HoldoutFile, 'rb') as i_file:
                HoldoutData = pickle.load(i_file)
            i_file.close()
            print('\n---- Load holdout data done. ----\n')

            variants = params['L1'][task]
            for variant in variants:
                print('\n==== Model inferring for level %s, algo %s, variant %s begins. ====\n' % ('L1', task, variant['variant']))
                best = variant['best']
                VarFields = variant['variant']
                varstr = ':'.join(['%s#%s' % (VarKey, VarFields[VarKey]) for VarKey in VarFields])
                score = best['mean']
                BestParams = best['params']
                print('\n---- Best parameter ---- \n')
                print(score, BestParams)
                print('\n-------- n')

                head = '%s:%s:%s' % ('L1', task, varstr)
                VarOutputDir = '%s/%s' % (OutputDir, head)
                if(os.path.exists(VarOutputDir) == False):
                    os.makedirs(VarOutputDir)
                model = d_model[task](BestParams, cls.kfold, InputDir, VarOutputDir)
                model.infer('%s:%s:%s' % ('L1', task, varstr), HoldoutData)
                print('\n==== Model inferring for level %s, algo %s, variant %s ends. ====\n' % ('L1', task, variant['variant']))
                del model
                gc.collect()

        return

    @classmethod
    def __run(cls, task, input, output):
        """"""
        modes = ['train', 'infer']
        HoldOutFile = '%s/holdout/test.pkl' % input

        for i in range(len(modes)):
            if (i == 0):
                InputDir = input
            else:
                InputDir = '%s/%s' % (output, modes[i - 1])
            OutputDir = '%s/%s' % (output, modes[i])
            if (os.path.exists(OutputDir) == False):
                os.makedirs(OutputDir)

            cls.__LaunchTask(task, modes[i], InputDir, OutputDir, HoldOutFile)

        return

    @classmethod
    def run(cls, tasks, InputDir, OutputDir):
        """"""
        for task in tasks:
            print('\n ==== Task for %s begins.  ====\n' % task)

            start = time.time()
            cls.__run(task, InputDir, OutputDir)
            end = time.time()

            print('\n ==== Task for %s done, time consumed %s. ====\n' % (task, (end - start)))

        return

