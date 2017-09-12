import pandas as pd
import abc
import os

class ModelBase(object):
    """"""
    __metaclass__  = abc.ABCMeta

    InputDir = ''
    OutputDir = ''
    data_format= 'pkl'
    parameters = {}
    kfold = 5

    def __init__(self, params, kfold, InputDir, OutputDir, data_format= 'pkl'):
        """"""
        ## initial
        self.InputDir = InputDir
        self.OutputDir = OutputDir
        self.parameters = params
        self.kfold = kfold
        self.data_format = data_format
        ## new output directory
        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

    @abc.abstractmethod
    def __fit(self):
        """"""

    @abc.abstractmethod
    def __predict(self):
        """"""

    @abc.abstractmethod
    def train(self, importance = False):
        """"""

    @abc.abstractmethod
    def infer(self, head, holdout, submit, metric_pk= False):
        """"""
