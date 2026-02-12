"""
Edge-GNP: Federated Learning Implementation
Module pour l'apprentissage fédéré avec élagage de graphes (federated_learning.py)

Auteur: Joshua Juste Emmanuel Yun Pei NIKIEMA
Cours: Algorithmics, Complexity, and Graph Algorithms
"""

import numpy as np
import networkx as nx
import torch
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import matplotlib.pyplot as plt

from graph_pruning import GraphPruner, GreedyEdgePruning
from gnn_model import GNNTrainer, networkx_to_pyg, Data


class FederatedClient:
    """
    Client pour l'apprentissage fédéré
    Chaque client possède son propre graphe local
    """
    
    def __init__(self,
                 client_id: int,
                 graph: nx.Graph,
                 node_features: np.ndarray,
                 labels: np.ndarray,
                 train_mask: np.ndarray,
                 val_mask: np.ndarray,
                 test_mask: np.ndarray,
                 pruner: Optional[GraphPruner] = None,
                 model_config: dict = None):
        """
        Args:
            client_id: Identifiant du client
            graph: Graphe local
            node_features: Caractéristiques des nœuds
            labels: Étiquettes
            train_mask, val_mask, test_mask: Masques pour les splits
            pruner: Algorithme d'élagage
            model_config: Configuration du modèle GNN
        """
        self.client_id = client_id
        self.original_graph = graph.copy()
        self.current_graph = graph.copy()
        self.node_features = node_features
        self.labels = labels
        
        # Convertir les masques en tenseurs PyTorch
        self.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        self.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        self.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        # Élagage
        self.pruner = pruner
        self.pruned_graph = None
        
        # Statistiques
        self.num_nodes = graph.number_of_nodes()
        self.num_edges_original = graph.number_of_edges()
        self.num_edges_current = graph.number_of_edges()
        
        # Modèle local
        if model_config is None:
            model_config = {
                'model_type': 'gcn',
                'num_features': node_features.shape[1],
                'hidden_dim': 64,
                'num_classes': len(np.unique(labels)),
                'num_layers': 2,
                'learning_rate': 0.01,
                'weight_decay': 5e-4
            }
        
        self.trainer = GNNTrainer(**model_config)
        
        # Convertir le graphe en format PyG
        self._update_data()
    
    def _update_data(self):
        """Met à jour l'objet Data PyTorch Geometric"""
        self.data = networkx_to_pyg(
            self.current_graph,
            self.node_features,
            self.labels
        )
    
    def prune_graph(self) -> Dict:
        """
        Élague le graphe local
        
        Returns:
            Statistiques de l'élagage
        """
        if self.pruner is None:
            return {
                'edges_before': self.num_edges_current,
                'edges_after': self.num_edges_current,
                'reduction_rate': 0.0
            }
        
        # Appliquer l'élagage
        self.pruned_graph = self.pruner.prune(self.current_graph, self.node_features)
        self.current_graph = self.pruned_graph
        self._update_data()
        
        # Statistiques
        edges_after = self.current_graph.number_of_edges()
        reduction = (self.num_edges_current - edges_after) / self.num_edges_current
        
        stats = {
            'edges_before': self.num_edges_current,
            'edges_after': edges_after,
            'reduction_rate': reduction
        }
        
        self.num_edges_current = edges_after
        
        return stats
    
    def local_training(self, num_epochs: int = 5) -> Dict:
        """
        Entraînement local sur le graphe (éventuellement élagué)
        
        Args:
            num_epochs: Nombre d'époques d'entraînement
            
        Returns:
            Statistiques d'entraînement
        """
        losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(num_epochs):
            # Entraînement
            loss = self.trainer.train_epoch(self.data, self.train_mask)
            losses.append(loss)
            
            # Évaluation
            train_acc, _ = self.trainer.evaluate(self.data, self.train_mask)
            val_acc, _ = self.trainer.evaluate(self.data, self.val_mask)
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
        
        return {
            'final_loss': losses[-1],
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'losses': losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def get_model_parameters(self) -> Dict:
        """Récupère les paramètres du modèle local"""
        return self.trainer.get_parameters()
    
    def set_model_parameters(self, parameters: Dict):
        """Met à jour les paramètres du modèle local"""
        self.trainer.set_parameters(parameters)
    
    def evaluate(self, mask_type: str = 'test') -> Tuple[float, float]:
        """
        Évalue le modèle
        
        Args:
            mask_type: 'train', 'val', ou 'test'
            
        Returns:
            (accuracy, loss)
        """
        if mask_type == 'train':
            mask = self.train_mask
        elif mask_type == 'val':
            mask = self.val_mask
        else:
            mask = self.test_mask
        
        return self.trainer.evaluate(self.data, mask)
    
    def get_communication_cost(self) -> int:
        """
        Calcule le coût de communication (nombre de paramètres + taille du graphe)
        
        Returns:
            Coût total en nombre d'éléments
        """
        # Nombre de paramètres du modèle
        num_params = sum(p.numel() for p in self.trainer.model.parameters())
        
        # Taille du graphe (arêtes)
        num_edges = self.current_graph.number_of_edges()
        
        return num_params + num_edges


class FederatedServer:
    """
    Serveur central pour l'apprentissage fédéré
    Agrège les modèles locaux via FedAvg
    """
    
    def __init__(self, model_config: dict):
        """
        Args:
            model_config: Configuration du modèle GNN global
        """
        self.model_config = model_config
        self.global_model = GNNTrainer(**model_config)
        self.round_history = []
    
    def get_global_parameters(self) -> Dict:
        """Récupère les paramètres du modèle global"""
        return self.global_model.get_parameters()
    
    def federated_averaging(self, 
                          client_parameters: List[Dict],
                          client_weights: List[float]) -> Dict:
        """
        Agrégation FedAvg
        
        w_global = Σ (n_i / n) * w_i
        
        Args:
            client_parameters: Liste des paramètres de chaque client
            client_weights: Poids de chaque client (proportionnel aux données)
            
        Returns:
            Paramètres agrégés
        """
        # Normaliser les poids
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Initialiser les paramètres agrégés
        aggregated_params = {}
        
        # Pour chaque paramètre
        for param_name in client_parameters[0].keys():
            # Moyenne pondérée
            aggregated_params[param_name] = sum(
                w * client_params[param_name]
                for w, client_params in zip(normalized_weights, client_parameters)
            )
        
        return aggregated_params
    
    def update_global_model(self, 
                          clients: List[FederatedClient],
                          selected_clients: Optional[List[int]] = None):
        """
        Met à jour le modèle global via FedAvg
        
        Args:
            clients: Liste de tous les clients
            selected_clients: Indices des clients sélectionnés (None = tous)
        """
        if selected_clients is None:
            selected_clients = list(range(len(clients)))
        
        # Récupérer les paramètres et poids
        client_params = []
        client_weights = []
        
        for idx in selected_clients:
            client = clients[idx]
            client_params.append(client.get_model_parameters())
            client_weights.append(client.num_nodes)  # Pondération par nombre de nœuds
        
        # Agrégation
        global_params = self.federated_averaging(client_params, client_weights)
        
        # Mise à jour du modèle global
        self.global_model.set_parameters(global_params)
    
    def broadcast_model(self, clients: List[FederatedClient]):
        """Diffuse le modèle global à tous les clients"""
        global_params = self.get_global_parameters()
        
        for client in clients:
            client.set_model_parameters(global_params)


class EdgeGNPFederated:
    """
    Système complet Edge-GNP pour apprentissage fédéré
    Intègre l'élagage de graphes et le GNN
    """
    
    def __init__(self,
                 clients: List[FederatedClient],
                 server: FederatedServer,
                 num_rounds: int = 50,
                 local_epochs: int = 5,
                 client_fraction: float = 1.0,
                 prune_every: int = 5):
        """
        Args:
            clients: Liste des clients
            server: Serveur fédéré
            num_rounds: Nombre de rounds de communication
            local_epochs: Nombre d'époques locales par round
            client_fraction: Fraction de clients sélectionnés par round
            prune_every: Fréquence d'élagage (en rounds)
        """
        self.clients = clients
        self.server = server
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction
        self.prune_every = prune_every
        
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'communication_cost': [],
            'pruning_stats': []
        }
    
    def select_clients(self, round_idx: int) -> List[int]:
        """
        Sélectionne un sous-ensemble de clients pour ce round
        
        Args:
            round_idx: Index du round
            
        Returns:
            Liste des indices de clients sélectionnés
        """
        num_clients = len(self.clients)
        num_selected = max(1, int(self.client_fraction * num_clients))
        
        # Sélection aléatoire
        np.random.seed(round_idx)
        selected = np.random.choice(num_clients, num_selected, replace=False)
        
        return selected.tolist()
    
    def run(self) -> Dict:
        """
        Exécute l'apprentissage fédéré Edge-GNP
        
        Returns:
            Historique de l'entraînement
        """
        print("="*70)
        print(" EDGE-GNP: FEDERATED LEARNING WITH GRAPH PRUNING")
        print("="*70)
        print(f"Clients: {len(self.clients)}")
        print(f"Rounds: {self.num_rounds}")
        print(f"Local Epochs: {self.local_epochs}")
        print(f"Client Fraction: {self.client_fraction}")
        print(f"Pruning Frequency: every {self.prune_every} rounds")
        print("="*70)
        
        for round_idx in range(self.num_rounds):
            print(f"\n{'='*70}")
            print(f"ROUND {round_idx + 1}/{self.num_rounds}")
            print(f"{'='*70}")
            
            # Sélectionner les clients
            selected_clients = self.select_clients(round_idx)
            print(f"Clients sélectionnés: {selected_clients}")
            
            # Diffuser le modèle global
            self.server.broadcast_model(self.clients)
            
            # Élagage périodique
            if round_idx % self.prune_every == 0 and round_idx > 0:
                print("\n[PRUNING PHASE]")
                pruning_stats = []
                
                for idx in selected_clients:
                    client = self.clients[idx]
                    stats = client.prune_graph()
                    pruning_stats.append(stats)
                    print(f"  Client {idx}: {stats['edges_after']} edges "
                          f"(reduction: {stats['reduction_rate']*100:.1f}%)")
                
                self.history['pruning_stats'].append({
                    'round': round_idx,
                    'stats': pruning_stats
                })
            
            # Entraînement local
            print("\n[LOCAL TRAINING]")
            for idx in selected_clients:
                client = self.clients[idx]
                train_stats = client.local_training(self.local_epochs)
                print(f"  Client {idx}: Loss={train_stats['final_loss']:.4f}, "
                      f"Train Acc={train_stats['final_train_acc']:.4f}, "
                      f"Val Acc={train_stats['final_val_acc']:.4f}")
            
            # Agrégation
            print("\n[AGGREGATION]")
            self.server.update_global_model(self.clients, selected_clients)
            
            # Diffuser le modèle mis à jour
            self.server.broadcast_model(self.clients)
            
            # Évaluation globale
            print("\n[GLOBAL EVALUATION]")
            avg_train_acc, avg_val_acc, avg_test_acc = self._evaluate_all_clients()
            
            print(f"  Average Train Acc: {avg_train_acc:.4f}")
            print(f"  Average Val Acc:   {avg_val_acc:.4f}")
            print(f"  Average Test Acc:  {avg_test_acc:.4f}")
            
            # Coût de communication
            total_comm_cost = sum(client.get_communication_cost() 
                                 for client in self.clients)
            print(f"  Communication Cost: {total_comm_cost:,} elements")
            
            # Enregistrer l'historique
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_acc'].append(avg_val_acc)
            self.history['test_acc'].append(avg_test_acc)
            self.history['communication_cost'].append(total_comm_cost)
        
        print("\n" + "="*70)
        print(" TRAINING COMPLETED")
        print("="*70)
        
        return self.history
    
    def _evaluate_all_clients(self) -> Tuple[float, float, float]:
        """
        Évalue tous les clients et retourne les moyennes
        
        Returns:
            (avg_train_acc, avg_val_acc, avg_test_acc)
        """
        train_accs = []
        val_accs = []
        test_accs = []
        
        for client in self.clients:
            train_acc, _ = client.evaluate('train')
            val_acc, _ = client.evaluate('val')
            test_acc, _ = client.evaluate('test')
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
        
        return np.mean(train_accs), np.mean(val_accs), np.mean(test_accs)
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualise les résultats de l'entraînement
        
        Args:
            save_path: Chemin pour sauvegarder la figure (optionnel)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Courbes d'accuracy
        rounds = range(1, len(self.history['train_acc']) + 1)
        
        axes[0].plot(rounds, self.history['train_acc'], label='Train', marker='o')
        axes[0].plot(rounds, self.history['val_acc'], label='Validation', marker='s')
        axes[0].plot(rounds, self.history['test_acc'], label='Test', marker='^')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Federated Learning Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Coût de communication
        axes[1].plot(rounds, self.history['communication_cost'], 
                    color='red', marker='d')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Communication Cost (elements)')
        axes[1].set_title('Communication Overhead')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Test du système Edge-GNP Fédéré")
    print("="*60)
    
    # Simuler 3 clients avec des graphes différents
    num_clients = 3
    clients = []
    
    for i in range(num_clients):
        # Créer un graphe aléatoire
        n = np.random.randint(30, 50)
        p = np.random.uniform(0.1, 0.3)
        G = nx.erdos_renyi_graph(n, p)
        
        # S'assurer que le graphe est connexe
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, p)
        
        # Caractéristiques aléatoires
        node_features = np.random.randn(n, 16)
        
        # Étiquettes binaires
        labels = np.random.randint(0, 2, n)
        
        # Splits train/val/test
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.6*n)]
        val_idx = indices[int(0.6*n):int(0.8*n)]
        test_idx = indices[int(0.8*n):]
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # Créer le client avec élagage
        pruner = GreedyEdgePruning(pruning_rate=0.2, importance_metric='betweenness')
        
        client = FederatedClient(
            client_id=i,
            graph=G,
            node_features=node_features,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            pruner=pruner
        )
        
        clients.append(client)
        print(f"Client {i}: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    
    # Créer le serveur
    model_config = {
        'model_type': 'gcn',
        'num_features': 16,
        'hidden_dim': 32,
        'num_classes': 2,
        'num_layers': 2,
        'learning_rate': 0.01
    }
    
    server = FederatedServer(model_config)
    
    # Créer et exécuter Edge-GNP
    edge_gnp = EdgeGNPFederated(
        clients=clients,
        server=server,
        num_rounds=20,
        local_epochs=3,
        client_fraction=1.0,
        prune_every=5
    )
    
    history = edge_gnp.run()
    
    # Visualiser les résultats
    edge_gnp.plot_results()
