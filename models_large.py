import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GINConv,GCNConv
from models_mp import GCN_MP, APPNP_MP, BernNet_MP
from linegraph import LineGraph2
import copy
from ChebNetII_pro_dual import ChebnetII_prop
import downsampling
from torch_geometric.data import Data

def comp_edge_weight(edge_index, q):
    """
    Draw a heatmap of the weight matrix of edges.
    :param edge_index: (2, 2E), E  the number of edges.
    :param q: (E, 1), E  the number of edges, q represent the weights of edges.
    :return:
        norm: (2E, 1)
    """
    norm = q.repeat(2,1)
    return edge_index, norm

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.mlp_layers = args.mlp_layers
        self.Original_ot = args.Original_ot
        self.activation = args.activation
        self.A_F = args.A_F
        self.Q = Parameter(torch.Tensor(int(dataset.data.edge_index.size(1) / 2), dataset.num_classes))

        if self.Original_ot == 'ot' and self.A_F:
            if self.mlp_layers == 2:
                self.lin1 = Linear(self.Q.size(1), 32)
                self.lin2 = Linear(32, 1)

        line_graph_trans = LineGraph2()
        dual_dataset  = line_graph_trans(copy.copy(self.dataset.data))
        self.dual_dataset = dual_dataset.detach()

        self.conv1 = GCN_MP(args, dataset, dataset.num_features, args.hidden)
        self.conv2 = GCN_MP(args, dataset, args.hidden, dataset.num_classes)
        self.gcn1_dual = GCNConv(4, args.hidden)
        self.gcn2_dual = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.kaiming_uniform_(self.Q)

    def forward(self, data, epoch):
        x, edge_index = data.x, data.edge_index
        if self.Original_ot == 'ot' and self.A_F:
            if self.mlp_layers == 2:
                if self.activation == 'sigmoid':
                    W = F.relu(self.lin1(self.Q))
                    W = self.lin2(W)
                    W = torch.sigmoid(W)
            edge_index, edge_weight = comp_edge_weight(edge_index, W)

            x = F.relu(self.conv1(data, epoch, x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(data, epoch, x, edge_index, edge_weight)
        else:
            x = F.relu(self.conv1(data, epoch, x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(data, epoch, x, edge_index)
        return [x, self.Q]

class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        #self.Q = Parameter(torch.Tensor(int(dataset.data.edge_index.size(1) / 2), dataset.num_classes))

        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout
        self.reset_parameters()
        self.dataset = dataset
        line_graph_trans = LineGraph2()
        dual_dataset  = line_graph_trans(copy.copy(self.dataset.data))
        self.dual_dataset = dual_dataset.detach()
        '''
        if args.dataset =='Squirrel': #downsampling
            kpool=KMISPooling(k=1)
            newdata=kpool(self.dual_dataset.x,self.dual_dataset.edge_index)[0]
            self.dual_dataset=Data(x=newdata.x,edge_index=newdata.edge_index,edge_attr=newdata.edge_attr)
        elif args.dataset =='Computers': #downsampling
            kpool=KMISPooling(k=3)
            newdata=kpool(self.dual_dataset.x,self.dual_dataset.edge_index)[0]
            self.dual_dataset=Data(x=newdata.x,edge_index=newdata.edge_index,edge_attr=newdata.edge_attr)
        #print(self.dual_dataset.edge_index.shape)
        
        if args.dataset in ['Computers','Squirrel']:
            if args.sample_linegraph: #sample linegraph
                if args.sample_method=='kmis':
                    pool=downsampling.KMISPooling(k=args.kmis_k)
                    newdata=pool(self.dual_dataset.x,self.dual_dataset.edge_index)[0]
                    self.dual_dataset=Data(x=newdata.x,edge_index=newdata.edge_index,edge_attr=newdata.edge_attr)
                elif args.sample_method=='random':
                    pool=downsampling.Random_EdgeSelection(p=args.random_p)
                    newdata=pool(self.dual_dataset.x,self.dual_dataset.edge_index,self.dual_dataset.edge_attr,self.dual_dataset.is_directed())
                    self.dual_dataset=Data(x=newdata.x,edge_index=newdata.edge_index,edge_attr=newdata.edge_attr)

        '''
        self.dual_dataset=self.dual_dataset.to(torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))
        
        #n=self.dual_dataset.num_node_features
        self.gcn1_dual = GCNConv(4, args.hidden)
        self.gcn2_dual = GCNConv(args.hidden, dataset.num_classes)


        




    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        #nn.init.kaiming_uniform_(self.Q)

    def forward(self, data, epoch):
        #print(self.dual_dataset.edge_index)
        self.Q1 =self.gcn1_dual(self.dual_dataset.x, self.dual_dataset.edge_index)
        #print(self.Q1.shape)
        self.Q=self.gcn2_dual(F.relu(self.Q1),self.dual_dataset.edge_index)
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return [x, self.Q]
    def predict(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

class GIN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GIN_Net, self).__init__()
        self.Q = Parameter(torch.Tensor(int(dataset.data.edge_index.size(1) / 2), dataset.num_classes))

        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(dataset.num_features, args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, args.hidden)
            ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(args.hidden, args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, dataset.num_classes)
            ))
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.kaiming_uniform_(self.Q)

    def forward(self, data, epoch):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return [x, self.Q]

class GSAGE_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GSAGE_Net, self).__init__()
        self.Q = Parameter(torch.Tensor(int(dataset.data.edge_index.size(1) / 2), dataset.num_classes))

        self.conv1 = SAGEConv(
            dataset.num_features,
            args.hidden)
        self.conv2 = SAGEConv(
            args.hidden,
            dataset.num_classes)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.kaiming_uniform_(self.Q)

    def forward(self, data, epoch):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return [x, self.Q]

class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP_MP(args, dataset, args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data, epoch):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(data, epoch, x, edge_index)
        return x

class BernNet(torch.nn.Module):
    def __init__(self,dataset, args):
        super(BernNet, self).__init__()

        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = BernNet_MP(args, args.K, dataset)
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data, epoch):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(data, epoch, x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(data, epoch, x, edge_index)
            return x

class ChebNetII(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = ChebnetII_prop(args, args.K, dataset)
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()



    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, epoch):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        return x
    
class ChebNetII(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = ChebnetII_prop(args, args.K, dataset)
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()



    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, epoch):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        return x

class ChebNetII_V(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNetII_V, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = ChebnetII_prop(args.K, True)

        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)



class ChebNetII_V(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNetII_V, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = ChebnetII_prop(args.K, True)

        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)

