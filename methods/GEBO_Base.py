import collections
import pickle
import random
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize

from utils.acquisition import AcquisitionOnSubspace, EI, UCB
from utils.ml_utils.models import GP
from methods.BaseBO import BaseBO
from utils.probability import *
from utils.graph_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import itertools
from itertools import combinations, permutations
import os.path


class GEBO_Base(BaseBO):

    def __init__(self, args, ard=False):
        super().__init__(args)
        self.acq_type = 'UCB'

        self.X = []
        self.Y = []
       
        # To check the best vals
        self.gp_bestvals = []
        
        self.ARD = ard
        
        self.iteration = None
        
        self.ht_recommedations = []

        self.model_hp = None
        self.default_lengthscale = 0.2
        
        self.gp_opt_method = 'multigrad'
        
        self.model_update_interval = 1
        
        self.name = None
        self.expert = None
                       
        self.exp_name = args.exp_name
        
        self.train_batch = 5
        
        self.lr = args.lr
        self.head_num = None
        self.feat_dim = None
        self.K = None
        self.encoder_type = None
        
        self.graph_node = None
        self.is_directed = None
        self.offset = None
        

    def runTrials(self, trials, budget, seed, saving_path):
        # Initialize mean_bestvals, stderr, hist
        n_working = trials
        self.saving_path = saving_path
        self.init_seed = seed
                
        best_vals = []
        reward_history = []
        for i in range(trials):

            print("Running trial: ", i)
            self.trial_num = i

            df= self.runOptim(budget=budget, seed=i)
            best_vals.append(df['best_value'])
            reward_history.append(df[['edge_list','reward', 'node_reward', 'contributed', 'selected']])
            self.save_progress_to_disk(best_vals, reward_history, saving_path, df)

        # Runoptim updates the ht_recommendation histogram
        self.best_vals = best_vals
        self.mean_best_vals = np.mean(best_vals, axis=0)
        self.err_best_vals = np.std(best_vals, axis=0) / np.sqrt(n_working)
        
        return self.mean_best_vals, self.err_best_vals

    def save_progress_to_disk(self, best_vals, reward_history, 
                              saving_path, df):
        results_file_name = saving_path + self.name + \
                            '_best_vals_' + \
                            self.acq_type + \
                            '_gp_opt_' + self.gp_opt_method + \
                            '_' + str(self.exp_name)

        with open(results_file_name, 'ab') as file:
            pickle.dump(best_vals, file)

        graph_reward_file_name = saving_path + self.name + \
                                    '_nheads_' + str(self.head_num) + \
                                    '_nheads_' + str(self.head_num) + \
                                    '_' + str(self.exp_name)
        
        with open(graph_reward_file_name, 'ab') as file:
            pickle.dump(reward_history, file)

        df.to_pickle(f"{results_file_name}_df_s{self.trial_num}")


    def update_weights_for_graphs(self, reward, gt, Wg_list, gamma,
                                       probabilityDistribution):
     
        estimatedReward = 1.0 * reward / probabilityDistribution[gt]
        Wg_list[gt] *= np.exp(estimatedReward * gamma / self.head_num)

        return Wg_list
       
    def update_weights_for_nodes(self, reward, hubnode_list, gt, Wn_list, gamma, 
                                 probabilityDistribution):
        hubnodes = hubnode_list[gt]
        estimatedReward = 1.0 * reward / probabilityDistribution[gt]
            
        for h in hubnodes:
            prob = 0
            for g, h_list in enumerate(hubnode_list):
                if h in h_list:
                    prob += probabilityDistribution[g]
            Wn_list[h] *= np.exp(estimatedReward * gamma / (prob*self.node*len(hubnodes)))

        return Wn_list

    def compute_prob_dist_and_draw_graph(self, Wc, gamma):
        probabilityDistribution = distr(Wc, gamma)
        gt = draw(probabilityDistribution)

        return gt, probabilityDistribution
           
           
    def sample_centered_graph(self, Wn, gamma, graphnum):
        hub_list = []
        graph_list = []
        probabilityDistribution_list = []
       
        for j in range(graphnum):
            probabilityDistribution = distr(Wn, gamma)
            hubs = set()
            for i in range(self.hubnum):
                h = draw(probabilityDistribution)
                hubs.add(h)
            hub_list.append(hubs)
            # create complete graph
            G = nx.Graph()
            G.add_edges_from(list(combinations(hubs, 2)))
            H = generate_graph(self.nDim, 1, hub=hubs, initial_graph=G)
            edges = np.array(H.edges).transpose(1,0)
            edges = np.hstack((edges, np.flip(edges,0)))
            graph_list.append(edges)
        probabilityDistribution_list.append(probabilityDistribution)
        return hub_list, graph_list, probabilityDistribution_list
    
    def train_autoencoder(self, gae, edge_index, data, result, device, batch, head_ind, h1=1, h2=1, K=1e-2):

        train_x = torch.Tensor(np.column_stack((data, result)))
        weight = 1/(np.array([(result < i).sum() for i in result]) + len(result)*K)
        weight /= sum(weight)
        weighted_sampler = WeightedRandomSampler(weight, self.train_batch, replacement=True)

        train_loader = DataLoader(TensorDataset(train_x, torch.Tensor(weight).to(device)), batch_size=batch, sampler=weighted_sampler)
        optimizer = torch.optim.Adam(gae.parameters(), lr=self.lr)

        gae.train()

        for data, w in train_loader:
            optimizer.zero_grad()
            warped_x, feat, out = gae.encode(data.to(device), edge_index, head_ind)
            pred, latent = gae.decode(out, edge_index[head_ind])

            loss = gae.recon_loss(warped_x, pred, w.to(device)) + 1/self.nDim * gae.kl_loss() + \
            h1*gae.metric_loss(out, data[:,-1].to(device)) + \
            h2*gae.encoder.orth_reg_loss(head_ind)

            loss.backward()
            optimizer.step()
          
        return gae.state_dict()
    
    # =============================================================================
    #     Over-ride this!
    # =============================================================================
    
    def runOptim(self, budget, seed):
        raise NotImplementedError

    # =============================================================================
    # Get best value from nested list along with the index
    # =============================================================================
    def getBestVal2(self, my_list):
        temp = [np.max(i * -1) for i in my_list]
        indx1 = [np.argmax(i * -1) for i in my_list]
        indx2 = np.argmax(temp)
        val = np.max(temp)
        list_indx = indx2
        val_indx = indx1[indx2]
        return val, list_indx, val_indx

    def set_model_params_and_opt_flag(self, model):
        """
        Returns opt_flag, model
        """
        if ((self.iteration >= self.model_update_interval) and
                (self.iteration % self.model_update_interval == 0)):
            return True, model
        else:
            # No previous model_hp, so optimise
            if self.model_hp is None:
                self.model_hp = model.param_array
            else:
                # print(self.model_hp)
                # print(model.param_array)
                # previous iter learned mix, so remove mix before setting
                if len(model.param_array) < len(self.model_hp):
                    model.param_array = self.model_hp[1:]
                else:
                    model.param_array = self.model_hp

            return False, model
 