import collections
import pickle
import random

import numpy as np
from utils.attention_module import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch_geometric
from torch_geometric.nn.models.autoencoder import VGAE
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from itertools import combinations, permutations
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor

MAX_LOGSTD = 10
    
class newGAE(VGAE):
    def __init__(self, encoder, decoder, hp_num, head_num, categorical_dims, bounds, device, dropout = 0.5, sep_ls=False, graph_node=False):
        super(newGAE, self).__init__(encoder, decoder)
        self.hp_num = hp_num
        self.feat_dim = encoder.feat_dim
        assert encoder.feat_dim == decoder.feat_dim, 'input and output feature dimension should match'
        self.categorical_dims = categorical_dims
        self.bounds = bounds
        self.device = device
        self.graph_node = graph_node
        
        self.mu = None
        self.logstd = None
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.prelu2 = nn.PReLU()
        
        self.embedding = nn.ModuleList()
        self.embedding2 = nn.ModuleList()
        for i in range(self.hp_num):
            if i in categorical_dims:
                self.embedding.append(nn.Linear(len(self.bounds[i]['domain']),self.feat_dim))
            else:
                self.embedding.append(nn.Linear(1, self.feat_dim))
            self.embedding2.append(nn.Linear(self.feat_dim, self.feat_dim))
        
        self.batch_norm = nn.BatchNorm1d(self.hp_num)
        
        self.head_num = head_num
        
        self.mha = MultiHeadAttention(self.head_num, self.feat_dim, self.feat_dim, self.feat_dim, self.hp_num, sep_ls=sep_ls)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def warp(self, x):
        """
        returns continuous value in unit scale;
        categorical in one-hot encoding
        """
        warped = [[] for _ in range(x.shape[0])]
        for n in range(x.shape[0]):
            for i in range(self.hp_num):
                if i not in self.categorical_dims:
                    # continuous
                    inp = x[n,i].unsqueeze(-1).to(self.device)
                    inp = (inp-self.bounds[i]['domain'][0])/(self.bounds[i]['domain'][1]-self.bounds[i]['domain'][0])
                else:
                    # categorical
                    src = torch.ones(len(self.bounds[i]['domain'])).to(self.device)
                    inp = torch.zeros(
                        len(self.bounds[i]['domain']), device=self.device
                    ).scatter_(0, x[n,i].to(torch.int64), src).to(self.device)
                warped[n].append(inp)
        return warped
        

    def encode(self, x, edge):
        # set categorical into one-hot encoding; continuous into unit scale 
        warped_x = self.warp(x)
        feature_matrix = torch.zeros(x.shape[0], self.hp_num, self.feat_dim).to(self.device)
        for n in range(x.shape[0]):
            for i in range(self.hp_num):
                feature_matrix[n,i,:] = self.prelu2(self.embedding2[i](
                    self.dropout(self.prelu(
                    self.embedding[i](warped_x[n][i])
                )))).unsqueeze(0)
        feature_matrix = self.batch_norm(feature_matrix)
        
        z_stack = []
        self.mu = []
        self.logstd = []
        # iterate through different subgraphs (heads) 
        for k in range(self.head_num):   
            structure_k = torch.Tensor(edge[k]).squeeze().to(torch.int64).to(self.device)
            self.__mu__, self.__logstd__ = self.encoder(feature_matrix, structure_k, k)
            z_k = self.reparametrize(self.__mu__, self.__logstd__)
            self.mu.append(self.__mu__)
            self.logstd.append(self.__logstd__)
            z_stack.append(z_k)
        
        if self.graph_node is False: 
            z = torch.stack(tuple(z_stack), dim=1) # shape : b * n_head * node_num * feat_dim
            graph_emb, attn = self.mha(z, z, z)
        else:
            graph_emb = []
            for i in range(self.head_num):
                graph_emb.append(z_stack[i][:,-1,:])
        return graph_emb
    
    
    def decode(self, z, h_i=0):
        """
        returns in list
        """
        output = self.decoder(z, h_i)
        return output
    
    def unwarp(self, output):
        """
        return in numpy array; to stack and optimize in LS
        """
        valid_output = torch.zeros(len(output), self.hp_num, 1).to(self.device)
        for n in range(len(output)):
            for i in range(self.hp_num):
                if i in self.categorical_dims:
                    out = torch.argmax(output[n][i], dim=1)
                else:
                    out = (torch.clamp(output[n][i],0,1) * (self.bounds[i]['domain'][1] - self.bounds[i]['domain'][0]) + self.bounds[i]['domain'][0]).reshape(-1, 1)
                valid_output[n,i,:] = out
        return valid_output.squeeze().detach().cpu().numpy()

    
    def recon_loss(self, x, z, label, head_ind=0):
        """Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges. 
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        # Frobenius Norm for node feature reconstruction
        # need to compute separately
        warped_x = self.warp(x)
        decoded = self.decode(z, head_ind)
        assert len(warped_x[0]) == len(decoded[0]), "the number of features don't match"
        total_loss = 0
        for n in range(len(label)):
            individual_loss = 0
            for i in range(len(warped_x[n])):
                individual_loss += torch.sum(torch.square(torch.abs(warped_x[n][i]-decoded[n][i])))
            total_loss += label[n]*torch.sqrt(individual_loss)/(2*sum(label))
        return total_loss
    
    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.
        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        if logstd is None:
            logstd = self.__logstd__
        else:
            for i in range(len(logstd)):
                logstd[i] = logstd[i].clamp(max=MAX_LOGSTD)
        
        if mu is not None and logstd is not None:
            loss = 0
            for i in range(len(mu)):
                loss += -0.5 * torch.mean(
                    torch.sum((1 + 2 * logstd[i] - mu[i]**2 - logstd[i].exp()**2), dim=1)
                )
            return loss
        
        else:
            print("one head?")
            return -0.5 * torch.mean(
                torch.sum((1 + 2 * logstd - mu**2 - logstd.exp()**2), dim=1)
            )


class Encoder(nn.Module): # from Kipf et al., VGAE encoder
    def __init__(self, feat_dim, device):
        super(Encoder, self).__init__()
        self.feat_dim = feat_dim
        self.device = device
        self.gc1 = GCNConv(feat_dim, feat_dim)
        self.relu1 = nn.ReLU()
        self.gc2a = GCNConv(feat_dim, feat_dim)
        self.gc2b = GCNConv(feat_dim, feat_dim)
        
    def forward(self,x, edge, ind):
        hidden1 = self.relu1(self.gc1(x, edge))
        mu = self.gc2a(hidden1, edge)
        logstd = self.gc2b(hidden1, edge)
        return mu, logstd
    
    def orth_reg_loss(self):
        """
        returns term for orthogonal regularization
        """
        loss = 0
        loss += torch.sum(torch.eye(self.gc1.weight.shape[0]).to(self.device) - torch.matmul(self.gc1.weight, self.gc1.weight.t()).to(self.device)) 
        return loss
    
class Decoder(nn.Module): 
    def __init__(self,feat_dim, bounds, categorical_dims, device, dropout=0.5):
        super(Decoder, self).__init__()
        self.feat_dim = feat_dim
        self.hp_num = len(bounds)
        self.bounds = bounds
        self.categorical_dims = categorical_dims
        self.device = device
        
#         self.layer = nn.ModuleList(
#             nn.Linear(self.feat_dim, self.feat_dim) 
#             for i in range(self.hp_num)
#         )

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.feat_dim, eps=1e-6)
        
        self.reverse_embedding = nn.ModuleList()
        for i in range(self.hp_num):
            if i in categorical_dims:
                self.reverse_embedding.append(nn.Linear(self.feat_dim,len(self.bounds[i]['domain'])))
            else:
                self.reverse_embedding.append(nn.Linear(self.feat_dim,1))
            nn.init.xavier_uniform_(self.reverse_embedding[-1].weight)

            
    def forward(self, z, h_i): # b * 1 * feat_dim
#         output = torch.zeros(z.shape[0], self.hp_num, 1).to(self.device)
        output = [[] for _ in range(z.shape[0])]
        for n in range(z.shape[0]):
            for i in range(self.hp_num):
                out = self.reverse_embedding[i](self.layer_norm(self.dropout(self.prelu(z[n,:,:])
                )))
                if i in self.categorical_dims:
                    out = nn.functional.softmax(out,dim=1)
                else:
                    pass
                output[n].append(out)
        return output