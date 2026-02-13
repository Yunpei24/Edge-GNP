"""
Edge-GNP: GNN Implementation
Module pour les réseaux de neurones graphiques (gnn_model.py)

Auteur: Joshua Juste Emmanuel Yun Pei NIKIEMA
Cours: Algorithmics, Complexity, and Graph Algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from typing import Optional, Tuple


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN)
    Implémentation du modèle de Kipf & Welling (2017)
    
    Équation de propagation:
    H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
    """
    
    def __init__(self, 
                 num_features: int,
                 hidden_dim: int = 64,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        """
        Args:
            num_features: Dimension des caractéristiques d'entrée
            hidden_dim: Dimension de la couche cachée
            num_classes: Nombre de classes (pour classification)
            num_layers: Nombre de couches GCN
            dropout: Taux de dropout
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Première couche
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        # Couches cachées
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Dernière couche
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, num_classes))
        else:
            self.convs = nn.ModuleList([GCNConv(num_features, num_classes)])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant
        
        Args:
            x: Matrice de caractéristiques [num_nodes, num_features]
            edge_index: Indices des arêtes [2, num_edges]
            
        Returns:
            Logits de sortie [num_nodes, num_classes]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Dernière couche sans activation
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Obtenir les embeddings avant la dernière couche
        
        Args:
            x: Caractéristiques des nœuds
            edge_index: Indices des arêtes
            
        Returns:
            Embeddings [num_nodes, hidden_dim]
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        return x


class GraphSAGE(nn.Module):
    """
    GraphSAGE: Inductive Representation Learning on Large Graphs
    Utilise l'agrégation de voisinage avec échantillonnage
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggregator: str = 'mean'):
        """
        Args:
            aggregator: Type d'agrégation ('mean', 'max', 'lstm')
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggregator))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggregator))
        else:
            self.convs = nn.ModuleList([SAGEConv(num_features, num_classes, aggr=aggregator)])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network (GAT)
    Utilise des mécanismes d'attention pour pondérer les voisins
    
    Attention: α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.5):
        """
        Args:
            num_heads: Nombre de têtes d'attention
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # Première couche avec multi-head attention
        self.convs.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Couches cachées
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                     heads=num_heads, dropout=dropout))
        
        # Dernière couche (moyenne des têtes)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * num_heads, num_classes, 
                                     heads=1, concat=False, dropout=dropout))
        else:
            self.convs = nn.ModuleList([GATConv(num_features, num_classes, 
                                               heads=1, concat=False, dropout=dropout)])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


def networkx_to_pyg(G: nx.Graph, 
                    node_features: Optional[np.ndarray] = None,
                    labels: Optional[np.ndarray] = None) -> Data:
    """
    Convertit un graphe NetworkX en objet PyTorch Geometric Data
    
    Args:
        G: Graphe NetworkX
        node_features: Matrice de caractéristiques [num_nodes, num_features]
        labels: Étiquettes des nœuds [num_nodes]
        
    Returns:
        Data object PyG
    """
    # Créer l'index des arêtes
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    
    # Ajouter les arêtes inverses pour graphe non-orienté
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Caractéristiques des nœuds
    if node_features is None:
        # Utiliser le degré comme caractéristique par défaut
        node_features = np.array([G.degree(node) for node in G.nodes()]).reshape(-1, 1)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Étiquettes
    y = None
    if labels is not None:
        y = torch.tensor(labels, dtype=torch.long)
    
    # Créer l'objet Data
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = G.number_of_nodes()
    
    return data


def train_gnn(model: nn.Module,
              data: Data,
              optimizer: torch.optim.Optimizer,
              criterion: nn.Module,
              train_mask: torch.Tensor,
              device: str = 'cpu') -> float:
    """
    Entraîne le GNN pour une époque
    
    Args:
        model: Modèle GNN
        data: Données du graphe
        optimizer: Optimiseur
        criterion: Fonction de perte
        train_mask: Masque des nœuds d'entraînement
        device: Device (cpu/cuda)
        
    Returns:
        Perte moyenne
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x.to(device), data.edge_index.to(device))
    
    # Calculer la perte uniquement sur les nœuds d'entraînement
    loss = criterion(out[train_mask], data.y[train_mask].to(device))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_gnn(model: nn.Module,
                data: Data,
                mask: torch.Tensor,
                device: str = 'cpu') -> Tuple[float, float]:
    """
    Évalue le GNN
    
    Args:
        model: Modèle GNN
        data: Données du graphe
        mask: Masque des nœuds à évaluer
        device: Device
        
    Returns:
        (accuracy, loss)
    """
    model.eval()
    
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)
        
        # Accuracy
        correct = (pred[mask] == data.y[mask].to(device)).sum()
        acc = int(correct) / int(mask.sum())
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out[mask], data.y[mask].to(device))
    
    return acc, loss.item()


class GNNTrainer:
    """Classe pour gérer l'entraînement des GNN"""
    
    def __init__(self,
                 model_type: str = 'gcn',
                 num_features: int = 1,
                 hidden_dim: int = 64,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 learning_rate: float = 0.01,
                 weight_decay: float = 5e-4,
                 device: str = 'cpu'):
        """
        Args:
            model_type: Type de modèle ('gcn', 'sage', 'gat')
        """
        self.device = device
        
        # Créer le modèle
        if model_type == 'gcn':
            self.model = GCN(num_features, hidden_dim, num_classes, num_layers)
        elif model_type == 'sage':
            self.model = GraphSAGE(num_features, hidden_dim, num_classes, num_layers)
        elif model_type == 'gat':
            self.model = GAT(num_features, hidden_dim, num_classes, num_layers)
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
        
        self.model = self.model.to(device)
        
        # Optimiseur
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, data: Data, train_mask: torch.Tensor) -> float:
        """Entraîne pour une époque"""
        return train_gnn(self.model, data, self.optimizer, 
                        self.criterion, train_mask, self.device)
    
    def evaluate(self, data: Data, mask: torch.Tensor) -> Tuple[float, float]:
        """Évalue le modèle"""
        return evaluate_gnn(self.model, data, mask, self.device)
    
    def get_parameters(self) -> dict:
        """Récupère les paramètres du modèle"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: dict):
        """Met à jour les paramètres du modèle"""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()
    
    def save_model(self, path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # Test du GNN
    print("Test du module GNN")
    
    # Créer un graphe de test
    G = nx.karate_club_graph()
    
    # Caractéristiques aléatoires
    num_nodes = G.number_of_nodes()
    node_features = np.random.randn(num_nodes, 16)
    
    # Étiquettes binaires aléatoires
    labels = np.random.randint(0, 2, num_nodes)
    
    # Convertir en PyG
    data = networkx_to_pyg(G, node_features, labels)
    print(f"Data: {data}")
    
    # Créer des masques train/val/test
    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)
    
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True
    
    # Entraîner un GCN
    trainer = GNNTrainer(
        model_type='sage',
        num_features=16,
        hidden_dim=32,
        num_classes=2,
        num_layers=2
    )
    
    print("\nEntraînement du GCN...")
    for epoch in range(50):
        loss = trainer.train_epoch(data, train_mask)
        
        if (epoch + 1) % 10 == 0:
            train_acc, _ = trainer.evaluate(data, train_mask)
            val_acc, val_loss = trainer.evaluate(data, val_mask)
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Test final
    test_acc, test_loss = trainer.evaluate(data, test_mask)
    print(f"\nTest Accuracy: {test_acc:.4f}")
