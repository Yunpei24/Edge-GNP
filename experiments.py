"""
Edge-GNP: Experiments and Benchmarks
Script d'expérimentation pour comparer les méthodes (experiments.py)

Auteur: Joshua Juste Emmanuel Yun Pei NIKIEMA
Cours: Algorithmics, Complexity, and Graph Algorithms
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import time
import json

from graph_pruning import (
    GreedyEdgePruning, 
    SpectralGraphSparsification, 
    CommunityAwarePruning,
    compare_pruning_methods
)
from gnn_model import GNNTrainer, networkx_to_pyg
from federated_learning import FederatedClient, FederatedServer, EdgeGNPFederated


class ExperimentSuite:
    """Suite d'expériences pour évaluer Edge-GNP"""
    
    def __init__(self, output_dir: str = "./results"):
        """
        Args:
            output_dir: Répertoire pour sauvegarder les résultats
        """
        self.output_dir = output_dir
        self.results = {}
    
    def generate_synthetic_graphs(self, 
                                  num_graphs: int = 5,
                                  graph_type: str = 'ba',
                                  min_nodes: int = 50,
                                  max_nodes: int = 200) -> List[nx.Graph]:
        """
        Génère des graphes synthétiques
        
        Args:
            num_graphs: Nombre de graphes à générer
            graph_type: Type de graphe ('ba', 'er', 'ws', 'sbm')
            min_nodes, max_nodes: Plage de taille
            
        Returns:
            Liste de graphes
        """
        graphs = []
        
        for i in range(num_graphs):
            n = np.random.randint(min_nodes, max_nodes)
            
            if graph_type == 'ba':  # Barabási-Albert (scale-free)
                m = np.random.randint(2, 5)
                G = nx.barabasi_albert_graph(n, m)
            
            elif graph_type == 'er':  # Erdős-Rényi
                p = np.random.uniform(0.05, 0.15)
                G = nx.erdos_renyi_graph(n, p)
                while not nx.is_connected(G):
                    G = nx.erdos_renyi_graph(n, p)
            
            elif graph_type == 'ws':  # Watts-Strogatz (small-world)
                k = np.random.randint(4, 10)
                p = np.random.uniform(0.1, 0.3)
                G = nx.watts_strogatz_graph(n, k, p)
            
            elif graph_type == 'sbm':  # Stochastic Block Model
                sizes = [n//3, n//3, n - 2*(n//3)]
                p_in = 0.3
                p_out = 0.05
                probs = [[p_in if i == j else p_out for j in range(3)] for i in range(3)]
                G = nx.stochastic_block_model(sizes, probs)
            
            else:
                raise ValueError(f"Type de graphe inconnu: {graph_type}")
            
            graphs.append(G)
        
        return graphs
    
    def experiment_pruning_rates(self, 
                                 G: nx.Graph,
                                 rates: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> Dict:
        """
        Expérience 1: Impact du taux d'élagage sur la performance
        
        Args:
            G: Graphe de test
            rates: Taux d'élagage à tester
            
        Returns:
            Résultats de l'expérience
        """
        print("\n" + "="*70)
        print("EXPÉRIENCE 1: Impact du Taux d'Élagage")
        print("="*70)
        
        results = {
            'rates': rates,
            'num_edges': [],
            'clustering': [],
            'diameter': [],
            'spectral_distance': [],
            'time': []
        }
        
        # Spectre original
        L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
        original_spectrum = np.linalg.eigvalsh(L)[:10]
        original_clustering = nx.average_clustering(G)
        original_diameter = nx.diameter(G) if nx.is_connected(G) else float('inf')
        
        for rate in rates:
            print(f"\nTaux d'élagage: {rate*100:.0f}%")
            
            # Greedy pruning
            pruner = GreedyEdgePruning(pruning_rate=rate, importance_metric='betweenness')
            
            start_time = time.time()
            G_pruned = pruner.prune(G.copy())
            elapsed_time = time.time() - start_time
            
            # Métriques
            num_edges = G_pruned.number_of_edges()
            clustering = nx.average_clustering(G_pruned)
            diameter = nx.diameter(G_pruned) if nx.is_connected(G_pruned) else float('inf')
            
            # Distance spectrale
            L_pruned = nx.normalized_laplacian_matrix(G_pruned).astype(float).toarray()
            pruned_spectrum = np.linalg.eigvalsh(L_pruned)[:10]
            spectral_dist = np.linalg.norm(original_spectrum - pruned_spectrum)
            
            results['num_edges'].append(num_edges)
            results['clustering'].append(clustering / original_clustering)
            results['diameter'].append(diameter / original_diameter if diameter != float('inf') else 1.0)
            results['spectral_distance'].append(spectral_dist)
            results['time'].append(elapsed_time)
            
            print(f"  Arêtes: {num_edges} ({num_edges/G.number_of_edges()*100:.1f}%)")
            print(f"  Clustering: {clustering:.4f} (ratio: {clustering/original_clustering:.4f})")
            print(f"  Distance spectrale: {spectral_dist:.6f}")
            print(f"  Temps: {elapsed_time:.3f}s")
        
        self.results['pruning_rates'] = results
        return results
    
    def experiment_compare_algorithms(self, 
                                     graphs: List[nx.Graph]) -> Dict:
        """
        Expérience 2: Comparaison des algorithmes d'élagage
        
        Args:
            graphs: Liste de graphes de test
            
        Returns:
            Résultats comparatifs
        """
        print("\n" + "="*70)
        print("EXPÉRIENCE 2: Comparaison des Algorithmes")
        print("="*70)
        
        algorithms = {
            'Greedy (Betweenness)': GreedyEdgePruning(pruning_rate=0.3, importance_metric='betweenness'),
            'Greedy (Similarity)': GreedyEdgePruning(pruning_rate=0.3, importance_metric='similarity'),
            'Community-Aware': CommunityAwarePruning(pruning_rate=0.3)
        }
        
        results = {alg_name: {
            'time': [],
            'edges_preserved': [],
            'clustering_ratio': [],
            'modularity': []
        } for alg_name in algorithms.keys()}
        
        for i, G in enumerate(graphs):
            print(f"\nGraphe {i+1}: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
            
            original_clustering = nx.average_clustering(G)
            
            for alg_name, pruner in algorithms.items():
                start_time = time.time()
                G_pruned = pruner.prune(G.copy())
                elapsed_time = time.time() - start_time
                
                # Métriques
                edges_ratio = G_pruned.number_of_edges() / G.number_of_edges()
                clustering = nx.average_clustering(G_pruned)
                clustering_ratio = clustering / original_clustering
                
                # Modularité
                try:
                    communities = nx.community.greedy_modularity_communities(G_pruned)
                    partition = {}
                    for idx, comm in enumerate(communities):
                        for node in comm:
                            partition[node] = idx
                    modularity = nx.community.modularity(G_pruned, communities)
                except:
                    modularity = 0.0
                
                results[alg_name]['time'].append(elapsed_time)
                results[alg_name]['edges_preserved'].append(edges_ratio)
                results[alg_name]['clustering_ratio'].append(clustering_ratio)
                results[alg_name]['modularity'].append(modularity)
                
                print(f"  {alg_name:25s}: {elapsed_time:.3f}s, "
                      f"Edges: {edges_ratio*100:.1f}%, "
                      f"Clustering: {clustering_ratio:.3f}")
        
        self.results['algorithm_comparison'] = results
        return results
    
    def experiment_federated_learning(self,
                                     num_clients: int = 5,
                                     num_rounds: int = 30,
                                     pruning_rates: List[float] = [0.0, 0.2, 0.4]) -> Dict:
        """
        Expérience 3: Apprentissage fédéré avec différents taux d'élagage
        
        Args:
            num_clients: Nombre de clients
            num_rounds: Nombre de rounds
            pruning_rates: Taux d'élagage à comparer
            
        Returns:
            Résultats d'apprentissage
        """
        print("\n" + "="*70)
        print("EXPÉRIENCE 3: Apprentissage Fédéré avec Élagage")
        print("="*70)
        
        results = {rate: {'history': None, 'final_acc': 0.0} for rate in pruning_rates}
        
        for rate in pruning_rates:
            print(f"\n{'='*70}")
            print(f"Taux d'élagage: {rate*100:.0f}%")
            print(f"{'='*70}")
            
            # Créer les clients
            clients = []
            for i in range(num_clients):
                # Graphe aléatoire
                G = nx.barabasi_albert_graph(60, 3)
                n = G.number_of_nodes()
                
                # Caractéristiques
                node_features = np.random.randn(n, 16)
                labels = np.random.randint(0, 2, n)
                
                # Splits
                train_mask = np.zeros(n, dtype=bool)
                val_mask = np.zeros(n, dtype=bool)
                test_mask = np.zeros(n, dtype=bool)
                
                indices = np.random.permutation(n)
                train_mask[indices[:int(0.6*n)]] = True
                val_mask[indices[int(0.6*n):int(0.8*n)]] = True
                test_mask[indices[int(0.8*n):]] = True
                
                # Pruner (seulement si rate > 0)
                pruner = GreedyEdgePruning(pruning_rate=rate) if rate > 0 else None
                
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
            
            # Serveur
            model_config = {
                'model_type': 'gcn',
                'num_features': 16,
                'hidden_dim': 32,
                'num_classes': 2,
                'num_layers': 2
            }
            server = FederatedServer(model_config)
            
            # Edge-GNP
            edge_gnp = EdgeGNPFederated(
                clients=clients,
                server=server,
                num_rounds=num_rounds,
                local_epochs=3,
                prune_every=5 if rate > 0 else 999
            )
            
            history = edge_gnp.run()
            
            results[rate]['history'] = history
            results[rate]['final_acc'] = history['test_acc'][-1]
            
            print(f"\nAccuracy finale: {history['test_acc'][-1]:.4f}")
        
        self.results['federated_learning'] = results
        return results
    
    def plot_all_results(self):
        """Visualise tous les résultats des expériences"""
        
        fig = plt.figure(figsize=(18, 10))
        
        # Expérience 1: Impact du taux d'élagage
        if 'pruning_rates' in self.results:
            data = self.results['pruning_rates']
            
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(data['rates'], data['num_edges'], marker='o', linewidth=2)
            ax1.set_xlabel('Taux d\'élagage')
            ax1.set_ylabel('Nombre d\'arêtes')
            ax1.set_title('Impact sur la Taille du Graphe')
            ax1.grid(True, alpha=0.3)
            
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(data['rates'], data['clustering'], marker='s', color='green', linewidth=2)
            ax2.set_xlabel('Taux d\'élagage')
            ax2.set_ylabel('Ratio de Clustering')
            ax2.set_title('Préservation du Clustering')
            ax2.grid(True, alpha=0.3)
            
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(data['rates'], data['spectral_distance'], marker='^', color='red', linewidth=2)
            ax3.set_xlabel('Taux d\'élagage')
            ax3.set_ylabel('Distance Spectrale')
            ax3.set_title('Préservation Spectrale')
            ax3.grid(True, alpha=0.3)
        
        # Expérience 2: Comparaison d'algorithmes
        if 'algorithm_comparison' in self.results:
            data = self.results['algorithm_comparison']
            
            ax4 = plt.subplot(2, 3, 4)
            for alg_name, metrics in data.items():
                ax4.plot(metrics['clustering_ratio'], label=alg_name, marker='o')
            ax4.set_xlabel('Graphe')
            ax4.set_ylabel('Ratio de Clustering')
            ax4.set_title('Comparaison: Préservation du Clustering')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            ax5 = plt.subplot(2, 3, 5)
            times = [np.mean(metrics['time']) for metrics in data.values()]
            ax5.bar(range(len(data)), times, tick_label=list(data.keys()))
            ax5.set_ylabel('Temps (s)')
            ax5.set_title('Temps d\'Exécution Moyen')
            ax5.tick_params(axis='x', rotation=15)
            ax5.grid(True, alpha=0.3, axis='y')
        
        # Expérience 3: Apprentissage fédéré
        if 'federated_learning' in self.results:
            data = self.results['federated_learning']
            
            ax6 = plt.subplot(2, 3, 6)
            for rate, results in data.items():
                if results['history']:
                    rounds = range(1, len(results['history']['test_acc']) + 1)
                    ax6.plot(rounds, results['history']['test_acc'], 
                            label=f'Pruning {rate*100:.0f}%', marker='o')
            ax6.set_xlabel('Round')
            ax6.set_ylabel('Test Accuracy')
            ax6.set_title('Apprentissage Fédéré: Impact de l\'Élagage')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_experiments.png', dpi=300, bbox_inches='tight')
        print(f"\nFigure sauvegardée: {self.output_dir}/all_experiments.png")
        plt.show()
    
    def save_results(self, filename: str = 'results.json'):
        """Sauvegarde les résultats en JSON"""
        # Convertir les éléments non-sérialisables
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        serializable_results = {}
        for exp_name, exp_data in self.results.items():
            if isinstance(exp_data, dict):
                serializable_results[exp_name] = {
                    k: convert(v) for k, v in exp_data.items()
                }
        
        filepath = f'{self.output_dir}/{filename}'
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Résultats sauvegardés: {filepath}")


def run_full_benchmark():
    """Exécute la suite complète d'expériences"""
    
    print("\n" + "="*70)
    print(" EDGE-GNP: SUITE COMPLÈTE D'EXPÉRIMENTATIONS")
    print("="*70)
    
    # Créer la suite d'expériences
    suite = ExperimentSuite(output_dir='./results')
    
    # Générer des graphes de test
    print("\nGénération des graphes de test...")
    test_graphs = suite.generate_synthetic_graphs(
        num_graphs=5,
        graph_type='ba',
        min_nodes=80,
        max_nodes=150
    )
    
    # Expérience 1: Impact du taux d'élagage
    suite.experiment_pruning_rates(
        G=test_graphs[0],
        rates=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    # Expérience 2: Comparaison d'algorithmes
    suite.experiment_compare_algorithms(graphs=test_graphs)
    
    # Expérience 3: Apprentissage fédéré (version réduite pour demo)
    suite.experiment_federated_learning(
        num_clients=3,
        num_rounds=15,
        pruning_rates=[0.0, 0.3]
    )
    
    # Visualiser et sauvegarder
    suite.plot_all_results()
    suite.save_results()
    
    print("\n" + "="*70)
    print(" EXPÉRIMENTATIONS TERMINÉES")
    print("="*70)


if __name__ == "__main__":
    run_full_benchmark()
