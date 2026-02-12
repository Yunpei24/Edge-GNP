# Edge-GNP: On-Device Graph Pruning for Communication-Efficient Federated Learning

## 📋 Description

Edge-GNP est un projet de recherche combinant l'apprentissage fédéré (Federated Learning) et l'élagage de graphes pour l'entraînement distribué de réseaux de neurones graphiques (GNN) sur des terminaux à ressources limitées.

**Auteur:** Votre Nom  
**Cours:** Algorithmics, Complexity, and Graph Algorithms

## 🎯 Objectifs

L'objectif principal est d'apprendre les paramètres **w** d'un GNN qui minimise la perte agrégée tout en respectant des contraintes de communication:

```
min_{w, {G̃_i}} F(w)  s.c.  C_comm ≤ B
```

où:
- **w**: Paramètres du GNN
- **G̃_i**: Graphes élagués des clients
- **F(w)**: Fonction de perte agrégée
- **C_comm**: Coût de communication
- **B**: Budget de communication

## 📁 Structure du Projet

```
Edge-GNP/
├── graph_pruning.py            # Algorithmes d'élagage de graphes
├── gnn_model.py                # Modèles GNN (GCN, GraphSAGE, GAT)
├── federated_learning.py       # Système d'apprentissage fédéré
├── experiments.py              # Suite d'expérimentations
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🔧 Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Installer PyTorch Geometric

**Pour CPU:**
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Pour GPU (CUDA 11.8):**
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 🚀 Utilisation

### Test des Algorithmes d'Élagage

```python
python graph_pruning.py
```

Ce script:
- Génère un graphe de test (Karate Club)
- Compare les 3 méthodes d'élagage:
  1. **Greedy Edge Pruning**: Élagage glouton basé sur l'importance
  2. **Spectral Sparsification**: Préservation du spectre du Laplacien
  3. **Community-Aware Pruning**: Préservation de la structure communautaire

### Test du GNN

```python
python gnn_model.py
```

Entraîne et évalue différents types de GNN:
- **GCN** (Graph Convolutional Network)
- **GraphSAGE**
- **GAT** (Graph Attention Network)

### Apprentissage Fédéré

```python
python federated_learning.py
```

Simule un système d'apprentissage fédéré avec:
- Plusieurs clients avec graphes locaux
- Élagage périodique des graphes
- Agrégation FedAvg
- Métriques de performance et communication

### Suite d'Expérimentations Complète

```python
python experiments.py
```

Exécute 3 expériences:
1. **Impact du taux d'élagage**: Évalue l'effet de ρ ∈ [0.1, 0.5]
2. **Comparaison d'algorithmes**: Compare les 3 méthodes d'élagage
3. **Apprentissage fédéré**: Teste différents taux d'élagage en FL

## 📊 Algorithmes Implémentés

### 1. Greedy Edge Pruning (GEP)

**Complexité:** O(ρm² + ρmn)

```python
pruner = GreedyEdgePruning(
    pruning_rate=0.3,
    importance_metric='betweenness'  # 'betweenness', 'similarity', 'degree'
)
G_pruned = pruner.prune(G)
```

**Métriques d'importance:**
- **Betweenness Centrality**: BC(e) = Σ σ_st(e)/σ_st
- **Jaccard Similarity**: Sim(u,v) = |N(u)∩N(v)|/|N(u)∪N(v)|
- **Degree Product**: I(u,v) = deg(u) × deg(v)

### 2. Spectral Graph Sparsification (SGS)

**Complexité:** O(mkn²) où k = nombre de valeurs propres

```python
pruner = SpectralGraphSparsification(
    pruning_rate=0.3,
    num_eigenvalues=10
)
G_pruned = pruner.prune(G)
```

Préserve les valeurs propres dominantes du Laplacien normalisé.

### 3. Community-Aware Pruning (CAP)

**Complexité:** O(m log m) avec détection Louvain

```python
pruner = CommunityAwarePruning(
    pruning_rate=0.3,
    preserve_intra=True
)
G_pruned = pruner.prune(G)
```

Détecte les communautés et préserve prioritairement les arêtes intra-communauté.

## 📈 Modèles GNN

### Graph Convolutional Network (GCN)

```python
model = GCN(
    num_features=16,
    hidden_dim=64,
    num_classes=2,
    num_layers=2,
    dropout=0.5
)
```

**Équation de propagation:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

### GraphSAGE

```python
model = GraphSAGE(
    num_features=16,
    hidden_dim=64,
    num_classes=2,
    aggregator='mean'  # 'mean', 'max', 'lstm'
)
```

### Graph Attention Network (GAT)

```python
model = GAT(
    num_features=16,
    hidden_dim=64,
    num_classes=2,
    num_heads=8
)
```

## 🔄 Apprentissage Fédéré

### Créer des Clients

```python
from federated_learning import FederatedClient

client = FederatedClient(
    client_id=0,
    graph=G,
    node_features=X,
    labels=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    pruner=GreedyEdgePruning(pruning_rate=0.3)
)
```

### Lancer Edge-GNP

```python
from federated_learning import EdgeGNPFederated, FederatedServer

# Serveur
server = FederatedServer(model_config)

# Système fédéré
edge_gnp = EdgeGNPFederated(
    clients=[client1, client2, client3],
    server=server,
    num_rounds=50,
    local_epochs=5,
    client_fraction=1.0,
    prune_every=5
)

# Entraînement
history = edge_gnp.run()

# Visualisation
edge_gnp.plot_results(save_path='results.png')
```


<!-- ## 🧪 Résultats Expérimentaux

Les expériences montrent:

1. **Réduction de communication:** 20-50% avec taux d'élagage ρ=0.3
2. **Préservation de performance:** ≥90% de l'accuracy originale
3. **Complexité:** Greedy est le plus rapide, Spectral le plus précis

### Exemple de Résultats

```
Taux d'élagage: 30%
- Arêtes conservées: 70%
- Test Accuracy: 0.89 (vs 0.92 sans élagage)
- Réduction communication: 35%
- Temps convergence: +10%
```

## 📊 Métriques Évaluées

- **Accuracy**: Précision de classification
- **Communication Cost**: Nombre de paramètres + arêtes transmis
- **Clustering Coefficient**: Préservation de la structure locale
- **Spectral Distance**: ||λ(L) - λ(L̃)||₂
- **Modularity**: Qualité de la structure communautaire
- **Training Time**: Temps par round -->

## 🔍 Analyse de Complexité

| Algorithme | Complexité Temps | Complexité Espace |
|------------|------------------|-------------------|
| Greedy Edge Pruning | O(ρm² + ρmn) | O(n + m) |
| Spectral Sparsification | O(mkn²) | O(n²) |
| Community-Aware | O(m log m) | O(n + m) |
| Edge-GNP (par round) | O(N·T_prune + N·E·T_GNN) | O(Np) |

où:
- **ρ**: Taux d'élagage
- **m**: Nombre d'arêtes
- **n**: Nombre de nœuds
- **k**: Nombre de valeurs propres
- **N**: Nombre de clients
- **E**: Époques locales
- **p**: Nombre de paramètres du modèle


## 🛠️ Développement Futur

- [ ] Élagage dynamique adaptatif
- [ ] Pruning différentiel pour confidentialité
- [ ] Support pour graphes hétérogènes
- [ ] Optimisation multi-objectifs
- [ ] Compression des paramètres GNN
- [ ] Benchmark sur datasets réels (Cora, CiteSeer, PubMed)

## 📧 Contact

Pour toute question sur le projet:
- **Email:** [Joshua.YUN-PEI@um6p.ma]
- **GitHub:** https://github.com/Yunpei24/Edge-GNP.git

## 📜 Licence

Ce projet est développé dans un cadre académique pour le cours "Algorithmics, Complexity, and Graph Algorithms".

## 🙏 Remerciements

- Professeur du cours Professeur Emerite Michel Habib pour les orientations
- Communauté PyTorch Geometric pour les outils GNN
- Travaux de recherche de McMahan et al. (FedAvg), Kipf & Welling (GCN)

---

**Note:** Ce projet est un prototype de recherche. Pour une utilisation en production, des optimisations supplémentaires et des tests de robustesse sont nécessaires.
