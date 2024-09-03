import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import GATConv, SAGEConv, TransformerConv
from torchvision.ops import sigmoid_focal_loss

# gnn models following deepsnap approach: https://github.com/snap-stanford/deepsnap/blob/master/examples/node_classification/node_classification_planetoid.py

class SAGEnorm(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, activation_function, loss_type='bce', alpha=0.25, gamma=2.0):
        super(SAGEnorm, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_layers = num_layers
        self.dropout = dropout
        self.loss_type = loss_type
        self.input_size = 300 # dim node features

        if activation_function == 'relu':
            self.activation = F.relu
        elif activation_function == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError("Unsupported activation function")

        self.convs = ModuleList()
        self.bns = ModuleList()  
        self.convs.append(SAGEConv(self.input_size, hidden_size)) 
        self.bns.append(BatchNorm1d(hidden_size))  

        for _ in range(1, self.num_layers):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
            self.bns.append(BatchNorm1d(hidden_size))  

        self.post_mp = Linear(hidden_size, 1)

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)  
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.post_mp(x)
        return x

    def loss(self, pred, label):
        label = label.float()
        pred = pred.squeeze(1)

        if self.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
            return criterion(pred, label)
        
        elif self.loss_type == "balanced_bce":
            #pos_weight = torch.tensor([self.alpha / (1 - self.alpha)]).to(label.device)
            pos_weight = torch.tensor([self.alpha]).to(label.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            return criterion(pred, label)
        
        elif self.loss_type == "focal":
            return sigmoid_focal_loss(pred, label, alpha=self.alpha, gamma=self.gamma, reduction="mean")           
        
        else:
            raise ValueError("Unsupported loss type")


class GATnorm(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, activation_function, num_heads, loss_type='bce', alpha=0.25, gamma=2.0):
        super(GATnorm, self).__init__()
        torch.manual_seed(22)
        self.alpha = alpha
        self.gamma = gamma
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.loss_type = loss_type
        self.input_size = 300

        if activation_function == 'relu':
            self.activation = F.relu
        elif activation_function == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError("Unsupported activation function")

        self.convs = ModuleList()
        self.bns = ModuleList()  
        self.convs.append(GATConv(self.input_size, hidden_size, heads=self.num_heads))
        self.bns.append(BatchNorm1d(hidden_size * self.num_heads))
        for _ in range(1, self.num_layers - 1):
            self.convs.append(GATConv(hidden_size * self.num_heads, hidden_size, heads=self.num_heads))
            self.bns.append(BatchNorm1d(hidden_size * self.num_heads))
        self.convs.append(GATConv(hidden_size * self.num_heads, hidden_size, heads=1, concat=False))
        self.bns.append(BatchNorm1d(hidden_size))  

        self.post_mp = Linear(hidden_size, 1)

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)  
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        x = self.bns[-1](x)  
        x = self.post_mp(x)
        return x

    def loss(self, pred, label):
        label = label.float()
        pred = pred.squeeze(1)

        if self.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
            return criterion(pred, label)
        
        elif self.loss_type == "balanced_bce":
            #pos_weight = torch.tensor([self.alpha / (1 - self.alpha)]).to(label.device)
            pos_weight = torch.tensor([self.alpha]).to(label.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            return criterion(pred, label)
        
        elif self.loss_type == "focal":
            return sigmoid_focal_loss(pred, label, alpha=self.alpha, gamma=self.gamma, reduction="mean")           
        
        else:
            raise ValueError("Unsupported loss type")


class GraphTransformernorm(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, activation_function, num_heads, loss_type="bce", alpha=0, gamma=0):
        super(GraphTransformernorm, self).__init__()
        torch.manual_seed(22)
        self.alpha = alpha
        self.gamma = gamma
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.loss_type = loss_type
        self.input_size = 300

        if activation_function == 'relu':
            self.activation = F.relu
        elif activation_function == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError("Unsupported activation function")

        self.convs = ModuleList()
        self.bns = ModuleList()  
        self.convs.append(TransformerConv(self.input_size, hidden_size, heads=self.num_heads))
        self.bns.append(BatchNorm1d(hidden_size * self.num_heads))
        for _ in range(1, self.num_layers - 1):
            self.convs.append(TransformerConv(hidden_size * self.num_heads, hidden_size, heads=self.num_heads))
            self.bns.append(BatchNorm1d(hidden_size * self.num_heads))
        self.convs.append(TransformerConv(hidden_size * self.num_heads, hidden_size, heads=1, concat=False))
        self.bns.append(BatchNorm1d(hidden_size))  

        self.post_mp = Linear(hidden_size, 1)

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)  
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index) #,return_attention_weights= True for interpret
        
        x = self.bns[-1](x)  
        x = self.post_mp(x)
        return x

    def loss(self, pred, label):
        label = label.float()
        pred = pred.squeeze(1)

        if self.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
            return criterion(pred, label)
        
        elif self.loss_type == "balanced_bce":
            #pos_weight = torch.tensor([self.alpha / (1 - self.alpha)]).to(label.device)
            pos_weight = torch.tensor([self.alpha]).to(label.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            return criterion(pred, label)
        
        elif self.loss_type == "focal":
            return sigmoid_focal_loss(pred, label, alpha=self.alpha, gamma=self.gamma, reduction="mean")           
        
        else:
            raise ValueError("Unsupported loss type")

