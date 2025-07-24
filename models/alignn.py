import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
import torch_scatter

class Standardize(nn.Module):
    """Standardize node features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: Data) -> Data:
        """Apply standardization to data.x (node features)."""
        data = data.clone()  # avoid modifying in-place
        data.x = (data.x - self.mean) / self.std
        return data

class RBFExpansion(nn.Module):
    def __init__(self, vmin=0, vmax=8, bins=40, lengthscale=None):
        super().__init__()
        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)
        self.gamma = 1 / ((lengthscale or (centers[1] - centers[0]).item()) ** 2)

    def forward(self, distance):
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class EdgeGatedGraphConvPyG(MessagePassing):
    """In AlIGNN paper the EdgeGatedGraphConvPyG is different from 
       gated conv layer from CGCNN
       (1) both edges and nodes are updated
       (2) Edges after update are turned into something like attention weight for the bond,
       play analog gates in CGCNN. Edge updated with 3 matrixes which are multiplied 
       by each of adjucent node feature and by the dge feature and then summed. In CGCNN
       one gate matrix is multiplied by the concatinated (not summed) features of nodes and edge
       (3) this implemented exactly as in the paper
    """
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__(aggr='add')
        self.residual = residual

        self.src_gate = nn.Linear(in_channels, out_channels)
        self.dst_gate = nn.Linear(in_channels, out_channels)
        self.edge_gate = nn.Linear(in_channels, out_channels)

        self.src_update = nn.Linear(in_channels, out_channels)
        self.dst_update = nn.Linear(in_channels, out_channels)

        self.bn_edges = nn.BatchNorm1d(out_channels)
        self.bn_nodes = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index):
        gate_input = self.src_gate(x_i) + self.dst_gate(x_j) + self.edge_gate(edge_attr)
        sigma = F.silu(gate_input)
        # Numerator
        msg = sigma * self.dst_update(x_j)
        # Denominator: sum σ_ij for each target node i
        norm = torch_scatter.scatter(sigma, index, dim=0, reduce="sum") + 1e-8
        norm = norm.index_select(0, index)  # expand to match msg shape
        return msg / norm
  
    def update(self, aggr_out, x):
        out = self.src_update(x) + aggr_out
        out = F.silu(self.bn_nodes(out))
        if self.residual:
            out += x
        return out


class ALIGNNConvPyG(nn.Module):
    """One ALIGNN layer composes an edge-gated graph convolution 
       on the bond graph (g) with an edge-gated graph convolution 
       on the line graph (L(g))
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_update = EdgeGatedGraphConvPyG(hidden_dim, hidden_dim)
        self.edge_update = EdgeGatedGraphConvPyG(hidden_dim, hidden_dim)

    def forward(self, data_g, data_lg):
        edge_attr = self.edge_update(data_lg.x, data_lg.edge_index, data_lg.edge_attr)
        x = self.node_update(data_g.x, data_g.edge_index, edge_attr)
        return x, edge_attr


class ALIGNN_PyG(nn.Module):
    """
    Initialize ALIGNN_PyG.

        Parameters
        ----------

        atom_input_features: int
          Number of atom features in the input.
        edge_input_features: int
          Number of bond features.
        triplet_input_features: int
          Number of angle features
        hidden_features: int
          Number of features in the internal representation
        alignn_layers: int
          Number of ALIGNN layers
        gcn_layers: int
          Number of EdgeGatedConv layers
        output_features: int
          The size of the output
    """
    def __init__(self, atom_input_features, hidden_features=64,
                 radius = 10.0, edge_input_features=40, triplet_input_features=20,
                 alignn_layers=4, gcn_layers=4,
                 classification=False, num_classes=2,
                 robust_regression=False, name='alignn'):
        super().__init__()
        self.name=name
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_input_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )
        self.edge_rbf = RBFExpansion(0, radius, edge_input_features)
        self.edge_embedding = nn.Sequential(
            self.edge_rbf,
            nn.Linear(edge_input_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )
        self.angle_rbf = RBFExpansion(-1.0, 1.0, triplet_input_features)
        self.angle_embedding = nn.Sequential(
            self.angle_rbf,
            nn.Linear(triplet_input_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )
        self.alignn_layers = nn.ModuleList([
            ALIGNNConvPyG(hidden_features) for _ in range(alignn_layers)
        ])
        self.gcn_layers = nn.ModuleList([
            EdgeGatedGraphConvPyG(hidden_features, hidden_features) for _ in range(gcn_layers)
        ])
        self.readout = global_mean_pool
        if classification:
            self.output_layer = nn.Linear(hidden_features, num_classes)
        elif robust_regression:
            self.output_layer = nn.Linear(hidden_features, 2)
        else:
            self.output_layer = nn.Linear(hidden_features, 1)

    def forward(self, data_g, data_lg):
        x = self.atom_embedding(data_g.x)
        edge_attr = self.edge_embedding(data_g.edge_attr)
        angle_attr = self.angle_embedding(data_lg.edge_attr)

        data_g.x = x
        data_g.edge_attr = edge_attr
        data_lg.x = edge_attr  # nodes in line graph = edges in G
        data_lg.edge_attr = angle_attr

        for layer in self.alignn_layers:
            x, edge_attr = layer(data_g, data_lg)
            data_g.x = x
            data_g.edge_attr = edge_attr
            data_lg.x = edge_attr

        for gcn in self.gcn_layers:
            x = gcn(data_g.x, data_g.edge_index, data_g.edge_attr)
            data_g.x = x

        x_pool = self.readout(x, data_g.batch)
        return self.output_layer(x_pool).squeeze(-1)