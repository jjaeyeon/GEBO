import os
import pickle
import random

import numpy as np

MAX_RANDOM_SEED = 128

class BaseBO():
    """
    Base class with common operations for BO with continuous and categorical
    inputs
    """

    def __init__(self, args):
        self.f = args.f  # function to optimise
        self.bounds = args.bounds  # function bounds
        self.C = args.categories  # no of categories
        self.initN = args.init_N  # no: of initial points
        self.nDim = len(self.bounds)  # dimension
        self.seed = args.seed
        self.saving_path = None
        self.x_bounds = np.vstack([d['domain'] for d in self.bounds
                                   if d['type'] == 'continuous'])
        
    def initialize(self, seed=0):
        data = []
        result = []

        initial_data_x = np.zeros((self.initN, self.nDim))
        seed_list = np.random.RandomState(seed).randint(0, MAX_RANDOM_SEED - 1, self.nDim)

        n_discrete = len(self.C)
        n_continuous = self.nDim - n_discrete
        
        for d in range(n_continuous):
            low, high = self.bounds[n_discrete + d]['domain']
            initial_data_x[:, d] = np.random.RandomState(seed_list[d]).uniform(low, high, self.initN)
        for d in range(n_discrete):
            domain = self.bounds[d]['domain']
            initial_data_x[:, d + n_continuous] = \
                np.random.RandomState(seed_list[d + n_continuous]).randint(0, len(domain), self.initN)

        Zinit = np.hstack([initial_data_x[:, n_continuous:], initial_data_x[:, :n_continuous]]).astype(np.float32)
        yinit = np.zeros([initial_data_x.shape[0], 1])

        for j in range(self.initN):
            ht_list = list(Zinit[j, :n_discrete])
            yinit[j] = self.f(ht_list, Zinit[j, n_discrete:])

        init_data = {}
        init_data['Z_init'] = Zinit
        init_data['y_init'] = yinit

        data.append(Zinit)
        result.append(yinit)
        print("initial data\n", data[0], "\nresult\n", result[0])
        
        return data, result

    def save_progress_to_disk(self, *args):
        raise NotImplementedError

    def runTrials(self, trials, budget, saving_path):
        raise NotImplementedError
