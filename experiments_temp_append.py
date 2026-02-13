
def run_experiment(model, data, pruner, args, 
                  run_classification=False, 
                  run_federated=False, 
                  run_pruning=False, 
                  run_comparison=False):
    """
    Exécute l'expérience demandée par main.py
    """
    print(f"\n{'='*60}")
    print(f"LANCEMENT DE L'EXPÉRIENCE: {args.experiment.upper()}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Classification Standard (Baseline)
    if run_classification or args.experiment == 'all':
        print("\n--- Classification Standard (Sans Élagage) ---")
        trainer = GNNTrainer(args.model, data.num_features, args.hidden_dim, 
                           dataset.num_classes if 'dataset' in locals() else 7, # Hack: Cora has 7 classes
                           args.num_layers, args.lr, args.weight_decay, device)
        
        # Override model with passed model if compatible (GNNTrainer creates its own)
        # Here we just use GNNTrainer's model for simplicity as it matches args
        
        # Train
        print(f"Entraînement sur {data.num_nodes} nœuds, {data.num_edges} arêtes...")
        for epoch in range(args.epochs):
            loss = trainer.train_epoch(data, data.train_mask)
            if (epoch+1) % 20 == 0:
                train_acc, _ = trainer.evaluate(data, data.train_mask)
                val_acc, _ = trainer.evaluate(data, data.val_mask)
                print(f"Epoch {epoch+1}: Loss {loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
        
        test_acc, _ = trainer.evaluate(data, data.test_mask)
        print(f"Test Accuracy (Original): {test_acc:.4f}")

    # 2. Pruning + Classification
    if run_pruning or args.experiment == 'all':
        print(f"\n--- Classification avec Élagage ({args.pruning_method}, rate={args.pruning_rate}) ---")
        
        # Convert PyG -> NetworkX
        from torch_geometric.utils import to_networkx
        G = to_networkx(data, to_undirected=True)
        print(f"Graphe original: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
        
        # Prune
        start_t = time.time()
        G_pruned = pruner.prune(G)
        prune_time = time.time() - start_t
        print(f"Temps d'élagage: {prune_time:.4f}s")
        print(f"Graphe élagué: {G_pruned.number_of_edges()} arêtes ({G_pruned.number_of_edges()/G.number_of_edges()*100:.1f}%)")
        
        # Convert NetworkX -> PyG
        # Attention: to_networkx perd les masques et features, il faut les remettre
        data_pruned = networkx_to_pyg(G_pruned, data.x.numpy(), data.y.numpy())
        data_pruned.train_mask = data.train_mask
        data_pruned.val_mask = data.val_mask
        data_pruned.test_mask = data.test_mask
        
        # Train on pruned
        trainer_p = GNNTrainer(args.model, data.num_features, args.hidden_dim, 
                             7, args.num_layers, args.lr, args.weight_decay, device)
                             
        for epoch in range(args.epochs):
            trainer_p.train_epoch(data_pruned, data_pruned.train_mask)
            
        test_acc_p, _ = trainer_p.evaluate(data_pruned, data_pruned.test_mask)
        print(f"Test Accuracy (Pruned): {test_acc_p:.4f}")
        
    # 3. Federated Learning
    if run_federated or args.experiment == 'all':
        print("\n--- Apprentissage Fédéré ---")
        suite = ExperimentSuite()
        suite.experiment_federated_learning(
            num_clients=args.num_clients,
            num_rounds=args.communication_rounds,
            pruning_rates=[0.0, args.pruning_rate]
        )

    # 4. Comparison
    if run_comparison:
        print("\n--- Comparaison des Algorithmes ---")
        from torch_geometric.utils import to_networkx
        G = to_networkx(data, to_undirected=True)
        compare_pruning_methods(G, args.pruning_rate, save_dir="Edge-GNP/images")
