import pandas as pd
import abc
import os

class ModelBase(object):
    """"""
    __metaclass__  = abc.ABCMeta

    InputDir = ''
    OutputDir = ''
    parameters = {}
    kfold = 5

    def __init__(self, params, kfold, InputDir,OutputDir):
        """"""
        ## initial
        self.InputDir = InputDir
        self.OutputDir = OutputDir
        self.parameters = params
        self.kfold = kfold
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
    def infer(self, head, holdout):
        """"""

    @abc.abstractmethod
    def submit(self):
        """"""
