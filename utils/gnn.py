import collections
import pickle
import random

import numpy as np

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
from torch_geometric.nn.norm import *


EPS = 1e-15
MAX_LOGSTD = 10


class GEBO_VGAE(VGAE):
    def __init__(self, encoder, decoder, hp_num, feat_dim, categorical_dims, bounds, device, dropout=0.5):
        super(GEBO_VGAE, self).__init__(encoder, decoder)
        self.hp_num = hp_num
        self.feat_dim = feat_dim
        assert self.feat_dim == encoder.feat_dim, 'feature dimension should match'
        self.bounds = bounds
        self.device = device
        self.categorical_dims = categorical_dims
        self.encoder = encoder
        self.decoder = decoder
        
        self.embedding = nn.ModuleList()
        for i in range(self.hp_num):
            if i in categorical_dims:
                self.embedding.append(nn.Linear(len(self.bounds[i]['domain']),self.feat_dim))
            else:
                self.embedding.append(nn.Linear(1, self.feat_dim))
        
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(self.feat_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def warp(self, x):
        """
        returns continuous value in unit scale;
        categorical in one-hot encoding; 
        returns in list; length equal to batch size
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
    
    def embed(self, x):
        """
        receiving input in list format
        """
        feature_matrix = torch.zeros(len(x), self.feat_dim, self.hp_num).to(self.device)
        for n in range(len(x)):
            for i in range(self.hp_num):
                feature_matrix[n,:,i] = self.embedding[i](x[n][i].unsqueeze(0))
        feature_matrix = self.relu(self.norm(feature_matrix))
        
        feature_matrix = torch.cat(
                (feature_matrix, torch.zeros(len(x),self.feat_dim,1).to(self.device)), dim=2
            )
        return feature_matrix
            
    def encode(self,x,edge,ind):
        warped_x = self.warp(x)
        feature_matrix = self.embed(warped_x).transpose(2,1)
        
        self.__mu__, self.__logstd__ = self.encoder(feature_matrix, edge, ind)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        
        graph_emb = z[:,-1,:].unsqueeze(1)
        
        return warped_x, feature_matrix, graph_emb
    
    def decode(self, z, edge):
        """
        returns in list
        """
        edge = torch.Tensor(edge).to(torch.int64).to(self.device)
        edge = edge[:,:-2*self.hp_num]
        pred, latent= self.decoder(z, edge)
        return pred, latent
    
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
                    out = (torch.clamp(output[n][i],0,1) * (
                        self.bounds[i]['domain'][1] - self.bounds[i]['domain'][0]
                    ) + self.bounds[i]['domain'][0]).reshape(-1, 1)
                valid_output[n,i,:] = out
        return valid_output.squeeze().detach().cpu().numpy()
    
    def metric_loss(self, graph_emb, perf):
        loss = 0
        for i in range(len(perf)):
            if (perf-perf[i]!=0).sum() == 0: #condition for exceptional case
                li = list(range(len(perf)))
                li.remove(i)
                pos_i = np.random.choice(li)
                li.remove(pos_i)
                neg_i = np.random.choice(li)
            else:
                pos_i = torch.argmin(abs(perf - perf[i])[perf-perf[i]!=0])
                if pos_i >= i:
                    pos_i += 1
                neg_i = torch.argmax(abs(perf - perf[i])[perf-perf[i]!=0])
                if neg_i >= i:
                    neg_i += 1

            loss += torch.square(
                torch.log(torch.sqrt(torch.sum(torch.square(graph_emb[i]-graph_emb[neg_i])))/torch.sqrt(torch.sum(torch.square(graph_emb[i]-graph_emb[pos_i])))) - \
                torch.log(torch.sqrt(torch.square(perf[i]-perf[neg_i]+1e-6))/torch.sqrt(torch.square(perf[i]-perf[pos_i]+1e-6)))
                                )
        loss /= len(perf)
        return loss
    
    def recon_loss(self,warped_x, pred, weight):
        assert len(warped_x[0]) == len(pred[0]), "the number of features don't match"
        total_loss = 0
        for n in range(len(pred)):
            individual_loss = 0
            for i in range(len(pred[n])):
                individual_loss += torch.sum(
                    torch.square(torch.abs(warped_x[n][i]-pred[n][i]))
                )
            total_loss += weight[n]*torch.sqrt(individual_loss)/2
        return total_loss

   
class Encoder(nn.Module): # separate gcn encoder for each head
    def __init__(self, feat_dim, head_num, hp_num, device):
        super(Encoder, self).__init__()
        self.feat_dim = feat_dim
        self.head_num = head_num
        self.hp_num = hp_num
        self.device = device 
        
        self.gc1 = nn.ModuleList(
            GCNConv(feat_dim, feat_dim)
            for _ in range(self.head_num)
        )
        
        self.relu = nn.ModuleList(
            nn.ReLU()
            for _ in range(self.head_num)
        )
        
        self.gc2a = nn.ModuleList(
            GCNConv(feat_dim, feat_dim)
            for _ in range(self.head_num)
        )
        
        self.gc2b = nn.ModuleList(
            GCNConv(feat_dim, feat_dim)
            for _ in range(self.head_num)
        )
        
    def forward(self,x, edge, ind):
        edge_index = edge[ind]
        
        # adding graph node
        from_all = np.arange(self.hp_num)
        dest = np.repeat(self.hp_num, self.hp_num)
        edge_index = np.hstack((edge_index, np.vstack((from_all, dest))))
        
        structure_k = torch.Tensor(edge_index).squeeze().to(torch.int64).to(self.device)
        hidden1 = self.relu[ind](self.gc1[ind](x, structure_k))
        mu = self.gc2a[ind](hidden1, structure_k)
        logstd = self.gc2b[ind](hidden1, structure_k)
        return mu, logstd

    
    def orth_reg_loss(self, ind):
        """
        returns term for orthogonal regularization
        """
        loss = 0
        for m in [self.gc1[ind], self.gc2a[ind], self.gc2b[ind]]:
            loss += torch.abs(torch.sum(torch.eye(m.weight.shape[0]).to(self.device) - torch.matmul(m.weight, m.weight.t()).to(self.device)))
        return loss

        return loss
    
class Decoder(nn.Module): 
    def __init__(self,feat_dim, bounds, categorical_dims, device, dropout=0.5):
        super(Decoder, self).__init__()
        self.feat_dim = feat_dim
        self.hp_num = len(bounds)
        self.bounds = bounds
        self.categorical_dims = categorical_dims
        self.device = device
        
        self.project = nn.ModuleList()
        self.reverse_embedding = nn.ModuleList()
        for i in range(self.hp_num):
            self.project.append(nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim)
            ))
            if i in categorical_dims:
                self.reverse_embedding.append(
                    nn.Linear(self.feat_dim,len(self.bounds[i]['domain']))
                )
            else:
                self.reverse_embedding.append(
                    nn.Linear(self.feat_dim,1)
                )
        self.norm1 = nn.LayerNorm([self.hp_num, self.feat_dim])
        self.relu1 = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, z, edge, sigmoid=True): # b * 1 * feat_dim
        pred = [[] for _ in range(z.shape[0])]
        latent = torch.zeros(z.shape[0], self.hp_num, self.feat_dim).to(self.device)
       
        for n in range(z.shape[0]):
            for i in range(self.hp_num):
                lat = self.project[i](z[n])
                latent[n,i,:] = lat
                
        latent = self.relu1(self.norm1(latent))
        
        for n in range(z.shape[0]):
            for i in range(self.hp_num):
                out = self.reverse_embedding[i](latent[n,i,:]).reshape(1,-1)
                if i in self.categorical_dims:
                    out = nn.functional.softmax(out,dim=1)
                else:
                    pass
                pred[n].append(out)    
                
        return pred, latent
   