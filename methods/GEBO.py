import math
import random
import time
import os

import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.acquisition import AcquisitionOnSubspace, EI, UCB
from utils.ml_utils.models import GP
from utils.ml_utils.optimization import sample_then_minimize
from utils.gnn import *

from methods.GEBO_Base import GEBO_Base
import torch_geometric
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
from itertools import combinations, permutations

class GEBO(GEBO_Base):

    def __init__(self, args):

        super(GEBO, self).__init__(args)
        self.best_val_list = []
        self.C_list = self.C
        self.name = 'GEBO'
        
        self.head_num = args.head_num
        self.feat_dim = args.ls_dimension
        self.K = args.K
        self.h1, self.h2 = list(map(float, args.loss_hp.split()))
        
        # graph sampling
        self.node = self.nDim
        self.hubnum = args.hub_num
        self.train_freq = 1
        self.duration = 5
        
        
    def runOptim(self, budget, seed, initData=None, initResult=None):

        if (initData and initResult):
            self.data = initData[:]
            self.result = initResult[:]
        else:
            self.data, self.result = self.initialize()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # initialize
        bestUpperBoundEstimate = 2 * budget / 3
        
        # graph MAB
        gamma = math.sqrt(self.head_num * math.log(self.head_num) /
                    ((math.e - 1) * bestUpperBoundEstimate))
        Wg_list_init = np.ones(self.head_num) 
        Wg_list = Wg_list_init
        
        # node MAB
        n_gamma = math.sqrt(self.node * math.log(self.node) /
                    ((math.e - 1) * bestUpperBoundEstimate))
        Wn_list_init = np.ones(self.node)
        Wn_list = Wn_list_init
        
        result_list = []
        starting_best, best_ind = np.max(-1 * self.result[0]), np.argmax(-1 * self.result[0])
        print(f"starting best point with value {starting_best}")
        
        print(', '.join(['%2d' % self.data[0][best_ind][i] for i in range(len(self.C))] +
                   ['%+.4f' % self.data[0][best_ind][i] for i in range(len(self.C), self.nDim)]))
        result_list.append([-1, None, starting_best, None])
        
        continuous_dims = list(range(len(self.C_list), self.nDim))
        categorical_dims = list(range(len(self.C_list)))
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        # model statement
        encoder = Encoder(self.feat_dim, self.head_num, self.nDim, device=device)
        decoder = Decoder(self.feat_dim, self.bounds, categorical_dims=categorical_dims, device=device)

        gae = GEBO_VGAE(encoder=encoder, 
                     decoder=decoder, 
                     hp_num=self.nDim, 
                     feat_dim = self.feat_dim,
                     categorical_dims=categorical_dims, 
                     bounds=self.bounds, 
                     device=device).to(device)

        hubnode_list, edge_index, pd_list = self.sample_centered_graph(Wn_list, n_gamma, self.head_num)
        
        print("initial graphs are")
        print(edge_index)
        
        restart_count = [0 for _ in range(self.head_num)]
        
        # warm up
        self.warmup_epoch = self.head_num
        for i in range(self.warmup_epoch):
            self.train_autoencoder(gae, edge_index, self.data[0], self.result[0], device, batch=2*self.train_batch, head_ind=i, h1=self.h1, h2=self.h2, K=1e+3)
        
        time_list = []
                 
        # optimization
        for t in tqdm(range(budget)):            
            self.iteration = t
            start = time.time()
            enhanced = False
            
            # compute prob over nodes
            # gt(len=1) returns which graph to encode
            gt, probabilityDistribution = \
                self.compute_prob_dist_and_draw_graph(Wg_list, gamma)       
            
            print("chose :", int(gt))
            
            gae.eval()

            next_to_evaluate, reward = self.lso_optimize(
                    gae, edge_index, self.f, gt, categorical_dims, continuous_dims, 
                    self.bounds, self.acq_type, device=device)
            
            Wg_list = self.update_weights_for_graphs(
                reward, gt,
                Wg_list, gamma,
                probabilityDistribution)
            
            Wn_list = self.update_weights_for_nodes(
                reward, hubnode_list, gt,
                Wn_list, n_gamma, probabilityDistribution)
            
            end = time.time()
            time_list.append(end-start)
            
            cat_list = next_to_evaluate[categorical_dims]
            # Get the best value till now
            besty, li, vi = self.getBestVal2(self.result)
            
            
            print("reward :", Wg_list)
            print("node reward :", Wn_list)

            print(', '.join(['%2d' % self.data[0][-1][i] for i in range(len(self.C))] +
                            ['%+.4f' % self.data[0][-1][i] for i in range(len(self.C), self.nDim)]))
            print("best val :", besty)
            
            if self.result[0][-1] > -besty:
                restart_count[gt] += 1
            else:
                enhanced = True
              
            # Store the results of this iteration
            result_list.append([t, besty, 
                                self.model_hp, 
                                edge_index.copy(), 
                                Wg_list.copy(), 
                                Wn_list.copy(), 
                                enhanced, 
                                int(gt)])
            
            # Conditional graph replacement
            if max(restart_count) > self.duration:
                graph_ind = np.argmax(restart_count)
                print(f"{graph_ind} Graph replaced ***********************")
                 # replace graph
                new_ht_list, new_graph = self.sample_centered_graph(Wn_list, n_gamma, 1)[:2]
                print(f"from \n {edge_index[graph_ind]} to \n {new_graph[0]}")
                edge_index[graph_ind] = new_graph[0]
                hubnode_list[graph_ind] = new_ht_list[0]
                Wg_list[graph_ind] = 1
                # normalize weight
                curr_sum = sum(Wg_list)-1
                Wg_list[:graph_ind] = Wg_list[:graph_ind]*((self.head_num-1)/curr_sum)
                Wg_list[graph_ind+1:] = Wg_list[graph_ind+1:]*((self.head_num-1)/curr_sum)
                # reset encoder
                for m in gae.encoder.modules():
                    if isinstance(m, nn.ModuleList):
                        if not isinstance(m[graph_ind], nn.ReLU):
                            m[graph_ind].reset_parameters()
                restart_count[graph_ind] = 0
           
            # Retraining
            if t % self.train_freq == 0 and t > 0:
                self.train_autoencoder(
                    gae, edge_index, self.data[0], self.result[0], 
                    device, batch=self.train_batch,
                    head_ind=int(gt), h1=self.h1, h2=self.h2, K=self.K
                )
            
        df = pd.DataFrame(result_list, columns=["iter", 
                                                "best_value", 
                                                "model_hp", "edge_list",
                                               "reward", "node_reward","contributed","selected"])
        bestx = self.data[li][vi]
        self.best_val_list.append([self.trial_num, li, besty,
                                   bestx])
        
        return df
    

    def lso_optimize(self, gae, edge, objfn, gt, categorical_dims,
                               continuous_dims, bounds, acq_type, device):
        """
        performs latent space optimization; GP-EI, GP-UCB
        """
        _,_,Xt = gae.encode(torch.Tensor(self.data[0]).to(device), edge, gt)
        Xt = Xt.squeeze().detach().cpu().numpy()
        yt = self.result[0]

        if self.ARD:
            hp_bounds = np.array([
                *[[1e-4, 3]] * self.feat_dim,  # lengthscale
                [1e-6, 1],  # likelihood variance
            ])
        else:
            hp_bounds = np.array([
                [1e-4, 3],  # lengthscale
                [1e-6, 1],  # likelihood variance
            ])
            
        kernel = GPy.kern.Matern52(self.feat_dim,
                                   lengthscale=np.sqrt(self.feat_dim), 
                                   ARD=self.ARD)
        

        gp_opt_params = {'method': self.gp_opt_method,
                         'num_restarts': 5,
                         'restart_bounds': hp_bounds,
                         'hp_bounds': hp_bounds,  
                         'verbose': False}
        
        gp = GP(Xt, yt, kernel, y_norm='meanstd', lik_variance_fixed = True,
                opt_params=gp_opt_params)
        
        opt_flag, gp = self.set_model_params_and_opt_flag(gp)
        if opt_flag:
            gp.optimize()
        self.model_hp = gp.param_array

        # graph embedding bounding
        length = np.std(Xt, axis=0) + 1e-6
        weights = kernel.lengthscale.item()
        lb = Xt - weights * length 
        ub = Xt + weights * length
        
        x_bounds = np.dstack((lb.min(axis=0),ub.max(axis=0))).squeeze(0)
        
        # create acq
        if acq_type == 'EI':
            acq = EI(gp, np.min(gp.Y_raw))
        elif acq_type == 'UCB':
            acq = UCB(gp, 2.0)
        
        
        def optimiser_func(x):
            return -acq.evaluate(np.atleast_2d(x))

        res = sample_then_minimize(
            optimiser_func,
            x_bounds,
            num_samples=5000,
            num_chunks=10,
            num_local=3,
            minimize_options=None,
            evaluate_sequentially=False)
        
        pred,_ = gae.decode(
            torch.Tensor(res.x).reshape(-1,1,self.feat_dim).to(device), edge[gt]
        )
        
        x_next = gae.unwarp(pred)
                
        x_next[categorical_dims] = x_next[categorical_dims].astype(np.int64)
        
        cont_next = x_next[continuous_dims]

        #  Evaluate objective function at z_next = [x_next,  ht_next_list]
        ht_next_list = x_next[categorical_dims]
        y_next = objfn(ht_next_list, cont_next)
                
        # Append recommeded data
        self.data[0] = np.row_stack((self.data[0], x_next))
        self.result[0] = np.row_stack((self.result[0], y_next))
        
        gt_reward = y_next
        
        return x_next, gt_reward
    
