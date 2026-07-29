from logging import config
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool,global_add_pool
from torch_scatter import scatter_std
#from alignn.alignn import data
from pydantic.typing import Literal
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from models.base import BaseSettings
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from models.utils import RBFExpansion

class BDIPNetConfig(BaseSettings):
    name: Literal["bdipnet","cgcnn","mono_cgcnn"]
    conv_layers: int = 3
    atom_input_features: int = 92
    inf_edge_features: int = 64
    fc_features: int = 256
    coulomb_vmin: float = -4.0
    coulomb_vmax: float = 4
    vdw_lj_vmin: float = -4
    vdw_lj_vmax: float = 4
    vdw_ld_vmin: float = -4
    vdw_ld_vmax: float = 4
    output_dim: int = 256
    output_features: int = 1
    rbf_min = -4.0
    rbf_max = 4.0
    potentials = []
    euclidean = False
    charge_map = False
    transformer = False

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"
 


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class PotNetConv(MessagePassing):

    def __init__(self, fc_features):
        super(PotNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.bn_edge = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )
        self.edge_update2 = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_i, x_j = x[row], x[col]

        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )

        return F.relu(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, index):
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))
    
class RelationSlotAttention(nn.Module):
    def __init__(self, fc_features):
        super().__init__()

        self.slot_att = nn.Sequential(
            nn.Linear(fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, 1),
        )
        self.bn_interaction = nn.BatchNorm1d(2)
        self.ln_interaction = nn.LayerNorm(2)
    def forward(self, x_c, x_v):
        # x_c, x_v: [N, F]
        slot_stack = torch.stack([x_c, x_v], dim=1)
        # [N, 2, F]

        score = self.slot_att(slot_stack)
        # [N, 2]
        score = score.squeeze(-1) 

        beta = torch.softmax(score, dim=1)
        
        # [N, 2]

        x_fused = (beta.unsqueeze(-1) * slot_stack).sum(dim=1)
        # [N, F]

        return x_fused, beta

class BDIPNetDualConv(MessagePassing):
    def __init__(self, fc_features):
        super().__init__(aggr="add", node_dim=0)
        
        self.fc_features = fc_features

        self.bn_coulomb = nn.BatchNorm1d(fc_features)
        self.bn_vdw = nn.BatchNorm1d(fc_features)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction_c = nn.BatchNorm1d(1)
        self.bn_interaction_v = nn.BatchNorm1d(1)

        # ===== Coulomb message MLP =====
        self.coulomb_nonlinear_full = nn.Sequential(
            nn.Linear(1 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, 1),
        )
        self.coulomb_nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )

        # ===== VDW message MLP =====
        self.vdw_nonlinear_full = nn.Sequential(
            nn.Linear(1 * fc_features,fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, 1),
        )
        self.vdw_nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )

        # ===== Attention trong message passing =====
        # x_all_i: 2F, x_all_j: 2F, edge_attr: F => 5F
        self.att_coulomb = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, 1),
        )

        self.att_vdw = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, 1),
        )
        self.slot_fuse = RelationSlotAttention(fc_features)

    def gated_message(
        self,
        x_i,
        x_j,
        edge_attr,
        nonlinear_full_net,
        nonlinear_net,
        mode,
        index
    ):
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        msg = nonlinear_net(z)
        if mode == "coulomb":
   
            score = nonlinear_full_net(msg)
      
            score = score.squeeze(-1) 
            alpha = softmax(score, index)
        elif mode == "vdw":
           
            score = nonlinear_full_net(msg)
       
            score = score.squeeze(-1) 
            alpha = softmax(score, index)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        
        return alpha.unsqueeze(-1) * msg
        #return gate * msg

    def forward(
                self,
                x,
                edge_index_col,
                edge_attr_col,
                edge_index_vdw,
                edge_attr_vdw,
                layer_index,
                is_homo=False,
                db="samba",
            ):
        Fdim = self.fc_features
        x_c = x
        x_v = x


        # ===== Coulomb slot =====
        msg_c = self.propagate(
            edge_index_col,
            x=x_c,
            edge_attr=edge_attr_col,
            mode="coulomb",

        )

        # ===== VDW slot =====
        if db != "c2db":
            msg_v = self.propagate(
                edge_index_vdw,
                x=x_v,
                edge_attr=edge_attr_vdw,
                mode="vdw",
            )
        else:
            msg_v = torch.zeros_like(msg_c)

        out, _ = self.slot_fuse(msg_c, msg_v)
    
        return F.relu(x + (out))

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        index,
        mode,
    ):
        # ===== Edge attention ====
        att_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        if mode == "coulomb":
            
           
         
            msg = self.gated_message(
                x_i,
                x_j,
                edge_attr,
                self.coulomb_nonlinear_full,
                self.coulomb_nonlinear,
                mode="coulomb",
                index=index
            )
    
        elif mode == "vdw":
        

            msg = self.gated_message(
                x_i,
                x_j,
                edge_attr,
                self.vdw_nonlinear_full,
                self.vdw_nonlinear,
                mode="vdw",
                index=index
            )
        
        else:
            raise ValueError(f"Unknown message mode: {mode}")

     
        return   msg
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing



class BDIPNet(nn.Module):

    def __init__(self, config: BDIPNetConfig = BDIPNetConfig(name="bdipnet")):
        super().__init__()
        self.config = config
        config.fc_features=128
        if not config.charge_map:
            self.atom_embedding = nn.Linear(
                config.atom_input_features, config.fc_features
            )
        else:
            self.atom_embedding = nn.Linear(
                config.atom_input_features + 10, config.fc_features
            )
        config.rbf_min=-4
        config.rbf_max=0
        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.fc_features,
                edge_type="coulomb",
                type='gaussian'
            ),
            nn.Linear(config.fc_features, config.fc_features),
            nn.SiLU(),
        )
       
       
        self.edge_embedding_vdw = nn.Sequential(
            RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.fc_features,
                edge_type="vdw",
                type='gaussian'
            ),
            nn.Linear(config.fc_features, config.fc_features),
            nn.SiLU(),
        )

       
        #self.weighted_norm_pool = WeightedNormalizedSumPool(config.fc_features)
        if not self.config.euclidean:
            self.inf_edge_embedding = RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.inf_edge_features,
                type='multiquadric'
            )

            self.infinite_linear = nn.Linear(config.inf_edge_features, config.fc_features)

            self.infinite_bn = nn.BatchNorm1d(config.fc_features)

        self.conv_layers = nn.ModuleList(
                    [
                        PotNetConv(config.fc_features)
                        for _ in range(config.conv_layers)
                    ]
                )

        
        self.conv_dual_layers = nn.ModuleList([BDIPNetDualConv(config.fc_features) for _ in range(config.conv_layers)])

        
        self.fc = nn.Sequential(
            nn.Linear(config.fc_features, config.fc_features), ShiftedSoftplus()
        )
        self.prop_fc = nn.Sequential(nn.Linear(1, config.fc_features))
        self.mv= nn.Linear(768, 256)
       
        self.slot_fuse = RelationSlotAttention(config.fc_features)
        self.fc_out = nn.Linear(config.fc_features, config.output_features)

        
    def forward(self, data,training=False, print_data=False):
        """CGCNN function mapping graph to outputs."""
        # fixed edge features: RBF-expanded bondlengths
        edge_index = data.edge_index.long()
        #print(edge_index)
        vdw=True
        node_features = self.atom_embedding(data.x)
        self.config.euclidean=True
        layer_index = data.layer_index
        is_homo = bool(data.is_homo_bilayer.all().item())
        if self.config.euclidean:        
            db="hetdb"
            if vdw==True:
             
                r = data.edge_attr
                dx = r[:,0]
                dy = r[:,1]
                dz = r[:,2]

                r_xyz = torch.sqrt(dx**2 + dy**2 + dz**2)
                edge_features = self.edge_embedding(-1/r_xyz)
               
                if db!="c2db":
                    r_vdw = data.r_vdw
                    dx = r_vdw[:,0]
                    dy = r_vdw[:,1]
                    dz = r_vdw[:,2]

                    r_vdw_xyz = torch.sqrt(dx**2 + dy**2 + dz**2)
                  
                    edge_vdw_features = self.edge_embedding_vdw((-1/(r_vdw_xyz**6)))
                  
                    edge_vdw_index = data.edge_vdw_index.long()
        
                else:
                    r_vdw = 0
                    x = 0
                    v_lj_vdw = 0
                    edge_vdw_features = 0
                    edge_vdw_index = 0
            else:
                r = data.edge_attr
                dx = r[:,0]
                dy = r[:,1]
                dz = r[:,2]

                r_xyz = torch.sqrt(dx**2 + dy**2 + dz**2)
                edge_features = self.edge_embedding(-1/r_xyz)
        else:

            edge_features = self.edge_embedding(-0.75 / data.edge_attr) 
           
            
        
        if not self.config.euclidean:
            inf_edge_index = data.inf_edge_index
            inf_feat = sum([data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.config.potentials)])
            inf_edge_features = self.inf_edge_embedding(inf_feat)
            inf_edge_features = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))

        # initial node features: atom feature network...
        if self.config.charge_map:
            node_features = self.atom_embedding(torch.cat([data.x, data.g_feats], -1))
       

       

        for i in range(self.config.conv_layers):
            if not self.config.euclidean and self.config.transformer:
                local_node_features = self.conv_layers[i](node_features, edge_index, edge_features)
                #inf_node_features = self.transformer_conv_layers[i](node_features, inf_edge_index, inf_edge_features)
                node_features = local_node_features #+ inf_node_features
            else:
               
                if vdw==True:
                    #node_features = self.conv_dual_layers[i](node_features,edge_index, edge_features,edge_vdw_index, edge_vdw_features,layer_index)
                    node_features = self.conv_dual_layers[i](
                            node_features,
                            edge_index,
                            edge_features,
                            edge_vdw_index,
                            edge_vdw_features,
                            layer_index,
                            is_homo=is_homo,
                            db=db,
                        )

                else:
                    node_features = self.conv_layers[i](node_features, edge_index, edge_features)
          
        mean_pool = global_mean_pool(node_features, data.batch)
      
        features = mean_pool
        features = self.fc(features)
        out = self.fc_out(features)
        return out.squeeze(-1)

