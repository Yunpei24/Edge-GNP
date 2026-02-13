"""
Main script for running experiments
"""

import argparse
import torch
from torch_geometric.datasets import Planetoid

from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

from gnn_model import GCN, GraphSAGE, GAT
from graph_pruning import GraphPruner, ModularAwarePruning

from experiments import run_experiment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Edge-GNP: Graph Neural Network Pruning')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'],
                        help='Dataset to use')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat'],
                        help='GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    # Pruning arguments
    parser.add_argument('--pruning_rate', type=float, default=0.5, help='Pruning rate')
    parser.add_argument('--pruning_method', type=str, default='modular', 
                        choices=['modular', 'random', 'degree', 'betweenness'],
                        help='Pruning method')
    
    # Federated learning arguments
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--communication_rounds', type=int, default=50, help='Communication rounds')
    parser.add_argument('--client_epochs', type=int, default=5, help='Epochs per client')
    
    # Experiment arguments
    parser.add_argument('--experiment', type=str, default='all', 
                        choices=['all', 'classification', 'federated', 'pruning', 'comparison'],
                        help='Experiment to run')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, default='results.png', help='Path to save results')
    
    return parser.parse_args()


def load_dataset(dataset_name: str):
    """Load dataset"""
    print(f"Loading dataset: {dataset_name}...")
    
    if dataset_name == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Dataset loaded: {dataset}")
    # Use standard train/val/test masks from Planetoid
    data = dataset[0]
    
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    return dataset, data


def select_model(model_name: str, num_features: int, num_classes: int):
    """Select GNN model"""
    if model_name == 'gcn':
        return GCN(num_features, hidden_dim=64, num_classes=num_classes)
    elif model_name == 'sage':
        return GraphSAGE(num_features, hidden_dim=64, num_classes=num_classes)
    elif model_name == 'gat':
        return GAT(num_features, hidden_dim=64, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def select_pruner(pruning_method: str, pruning_rate: float):
    """Select pruning method"""
    if pruning_method == 'modular':
        return ModularAwarePruning(pruning_rate)
    elif pruning_method == 'random':
        return GraphPruner(pruning_rate)
    elif pruning_method == 'degree':
        return GraphPruner(pruning_rate, method='degree')
    elif pruning_method == 'betweenness':
        return GraphPruner(pruning_rate, method='betweenness')
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")


def main():
    """Main function"""
    args = parse_args()
    
    # Load dataset
    dataset, data = load_dataset(args.dataset)
    
    # Select model
    model = select_model(args.model, dataset.num_features, dataset.num_classes)
    
    # Select pruner
    pruner = select_pruner(args.pruning_method, args.pruning_rate)
    
    # Run experiments
    if args.experiment == 'all':
        run_experiment(model, data, pruner, args)
    else:
        # Run specific experiment
        if args.experiment == 'classification':
            run_experiment(model, data, pruner, args, run_classification=True)
        elif args.experiment == 'federated':
            run_experiment(model, data, pruner, args, run_federated=True)
        elif args.experiment == 'pruning':
            run_experiment(model, data, pruner, args, run_pruning=True)
        elif args.experiment == 'comparison':
            run_experiment(model, data, pruner, args, run_comparison=True)
    
    print("Experiment completed!")


if __name__ == '__main__':
    main()