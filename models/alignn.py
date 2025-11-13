import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


class EdgeGatedGraphConvPyG(MessagePassing):
    """Edge-gated graph convolution layer with corrected message normalization."""
    
    def __init__(self, in_channels, out_channels, residual=True, use_layer_norm=True):
        super().__init__(aggr='add')
        self.residual = residual
        self.use_layer_norm = use_layer_norm

        self.src_gate = nn.Linear(in_channels, out_channels)
        self.dst_gate = nn.Linear(in_channels, out_channels)
        self.edge_gate = nn.Linear(in_channels, out_channels)

        self.src_update = nn.Linear(in_channels, out_channels)
        self.dst_update = nn.Linear(in_channels, out_channels)

        if use_layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Compute gate: element-wise combination of source, dest, and edge features
        gate = self.src_gate(x_i) + self.dst_gate(x_j) + self.edge_gate(edge_attr)
        gate = F.silu(gate)
        
        # Message: gated update from destination node
        msg = gate * self.dst_update(x_j)
        return msg
  
    def update(self, aggr_out, x):
        # Update: source node feature + aggregated messages
        out = self.src_update(x) + aggr_out
        
        # Normalize
        out = self.norm(out)
        out = F.silu(out)
        
        # Residual connection
        if self.residual and x.shape == out.shape:
            out = out + x
        
        return out


class ALIGNNConvPyG(nn.Module):
    """One ALIGNN layer: edge updates on line graph, then node updates on bond graph."""
    
    def __init__(self, hidden_dim, use_layer_norm=True):
        super().__init__()
        self.node_update = EdgeGatedGraphConvPyG(hidden_dim, hidden_dim, use_layer_norm=use_layer_norm)
        self.edge_update = EdgeGatedGraphConvPyG(hidden_dim, hidden_dim, use_layer_norm=use_layer_norm)

    def forward(self, data_g, data_lg):
        # Update edges using line graph
        edge_attr = self.edge_update(data_lg.x, data_lg.edge_index, data_lg.edge_attr)
        
        # Update nodes using updated edges
        x = self.node_update(data_g.x, data_g.edge_index, edge_attr)
        
        return x, edge_attr


class RBFExpansion(nn.Module):
    """Radial basis function expansion."""
    
    def __init__(self, vmin=0, vmax=8, bins=40, lengthscale=None):
        super().__init__()
        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)
        if lengthscale is None:
            lengthscale = (centers[1] - centers[0]).item()
        self.gamma = 1.0 / (lengthscale ** 2)

    def forward(self, distance):
        # RBF: exp(-gamma * (distance - center)^2)
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class Standardize(nn.Module):
    """Standardize node features."""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: Data) -> Data:
        data = data.clone()
        data.x = (data.x - self.mean) / (self.std + 1e-8)
        return data


class ALIGNN_PyG(nn.Module):
    """ALIGNN model for materials property prediction."""

    def __init__(self, 
                 atom_input_features,
                 hidden_features=64,
                 radius=10.0,
                 edge_input_features=40,
                 triplet_input_features=20,
                 alignn_layers=4,
                 gcn_layers=4,
                 classification=False,
                 num_classes=2,
                 robust_regression=False,
                 quantile_regression=False,
                 num_quantiles=1,
                 name='alignn',
                 use_layer_norm=True,
                 additional_compound_features=False,
                 add_feat_len=64):
        super().__init__()
        self.name = name
        self.hidden_features = hidden_features
        self.additional_compound_features = additional_compound_features
        
        if self.additional_compound_features:
            self.add_feat_len = add_feat_len

        # Atom embedding
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_input_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layer_norm else nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )

        # Edge (bond distance) embedding
        self.edge_rbf = RBFExpansion(0, radius, edge_input_features)
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layer_norm else nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )

        # Angle (triplet) embedding
        self.angle_rbf = RBFExpansion(-1.0, 1.0, triplet_input_features)
        self.angle_embedding = nn.Sequential(
            nn.Linear(triplet_input_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layer_norm else nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )

        # ALIGNN layers
        self.alignn_layers = nn.ModuleList([
            ALIGNNConvPyG(hidden_features, use_layer_norm=use_layer_norm) 
            for _ in range(alignn_layers)
        ])

        # Final GCN layers
        self.gcn_layers = nn.ModuleList([
            EdgeGatedGraphConvPyG(hidden_features, hidden_features, use_layer_norm=use_layer_norm) 
            for _ in range(gcn_layers)
        ])

        # Readout
        self.readout = global_mean_pool
        
        # Additional compound features processing
        if self.additional_compound_features:
            norm_layer = nn.LayerNorm(add_feat_len) if use_layer_norm else nn.BatchNorm1d(add_feat_len)
            self.add_feat_norm = norm_layer
            self.proj_add_feat = nn.Linear(add_feat_len, hidden_features)
            self.add_feat_activation = nn.SiLU()
            self.conv_to_fc = nn.Linear(2 * hidden_features, hidden_features)
        else:
            self.conv_to_fc = nn.Linear(hidden_features, hidden_features)

        self.fc_activation = nn.SiLU()

        # Output layer
        if classification:
            self.output_layer = nn.Linear(hidden_features, num_classes)
        elif robust_regression:
            self.output_layer = nn.Linear(hidden_features, 2)
        elif quantile_regression:
            self.output_layer = nn.Linear(hidden_features, num_quantiles)
        else:
            self.output_layer = nn.Linear(hidden_features, 1)

    def forward(self, data_g, data_lg):
        # Embed features
        x = self.atom_embedding(data_g.x)
        
        # Apply RBF expansion first, then embedding
        edge_attr = self.edge_rbf(data_g.edge_attr)
        edge_attr = self.edge_embedding(edge_attr)
        
        angle_attr = self.angle_rbf(data_lg.edge_attr)
        angle_attr = self.angle_embedding(angle_attr)

        # Set graph data
        data_g.x = x
        data_g.edge_attr = edge_attr
        data_lg.x = edge_attr  # nodes in line graph = edges in original graph
        data_lg.edge_attr = angle_attr

        # ALIGNN layers
        for layer in self.alignn_layers:
            x, edge_attr = layer(data_g, data_lg)
            data_g.x = x
            data_g.edge_attr = edge_attr
            data_lg.x = edge_attr

        # Final GCN layers
        for gcn in self.gcn_layers:
            x = gcn(data_g.x, data_g.edge_index, data_g.edge_attr)
            data_g.x = x

        # Readout and output
        x_pool = self.readout(x, data_g.batch)
        
        if self.additional_compound_features:
            add_feat = self.add_feat_norm(data_g.additional_compound_features)
            add_feat = self.proj_add_feat(add_feat)
            add_feat = self.add_feat_activation(add_feat)
            combined = torch.cat([x_pool, add_feat], dim=1)
            x_pool = self.conv_to_fc(combined)
        else:
            x_pool = self.conv_to_fc(x_pool)
        
        x_pool = self.fc_activation(x_pool)
        output = self.output_layer(x_pool).squeeze(-1)
        
        return output