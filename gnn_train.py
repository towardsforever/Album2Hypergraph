#!/usr/bin/env python3
"""
Simple edge-classification GNN baseline for social relation labels.

Data source: triangle_relations_corrected.txt (a b c r12 r23 r13).
We aggregate all edges (a,b), (b,c), (a,c) with majority vote on labels,
build an undirected graph, and train a GraphSAGE-style model to predict
the relation label for each edge.

Dependencies: torch, torch_geometric.
Run example:
  python3 tools/gnn_train.py \
    --triangle-file data/train_test_split/triangle_relations_corrected.txt \
    --epochs 30 --batch-size 1024
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


Edge = Tuple[int, int, int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_triangle_edges(path: Path, add_noise: bool = False) -> List[Edge]:
    """Load triangles and collapse to edge labels by majority vote per unordered pair.
    
    Args:
        path: 三角关系文件路径
        add_noise: 是否添加噪声
    """
    pair_labels: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        a, b, c, r12, r23, r13 = map(int, line.split())
        for u, v, r in ((a, b, r12), (b, c, r23), (a, c, r13)):
            key = (u, v) if u <= v else (v, u)
            pair_labels[key][r] += 1

    edges: List[Edge] = []
    for (u, v), cnt in pair_labels.items():
        label, _ = cnt.most_common(1)[0]
        edges.append((u, v, label))
    
    # 添加噪声
    if add_noise:
        num_edges = len(edges)
        num_noise = int(num_edges * 0.1)  # 添加10%的噪声
        for i in range(num_noise):
            # 随机选择一条边
            idx = random.randint(0, num_edges - 1)
            u, v, original_label = edges[idx]
            # 随机选择一个不同的标签
            new_label = random.randint(0, 15)  # 16种关系
            while new_label == original_label:
                new_label = random.randint(0, 15)
            edges[idx] = (u, v, new_label)
    
    return edges


def build_graph(edges: List[Edge]) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    """Construct PyG Data and edge label tensors with train/val/test splits."""
    node_ids = sorted({x for e in edges for x in (e[0], e[1])})
    id_map = {node: i for i, node in enumerate(node_ids)}

    # Build undirected edge_index
    edge_index_list = []
    edge_labels = []
    for u, v, lbl in edges:
        ui, vi = id_map[u], id_map[v]
        edge_index_list.append((ui, vi))
        edge_index_list.append((vi, ui))
        edge_labels.append(lbl)
        edge_labels.append(lbl)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(edge_labels, dtype=torch.long)

    # For supervision we keep only one direction per pair to avoid double counting
    sup_edge_pairs = [(id_map[u], id_map[v]) for u, v, _ in edges]
    sup_edge_index = torch.tensor(sup_edge_pairs, dtype=torch.long).t().contiguous()
    sup_edge_label = torch.tensor([lbl for _, _, lbl in edges], dtype=torch.long)

    data = Data(edge_index=edge_index, num_nodes=len(node_ids))
    return data, sup_edge_index, sup_edge_label


class EdgeDataset(Dataset):
    def __init__(self, edge_index: torch.Tensor, edge_label: torch.Tensor, indices: List[int]):
        self.edge_index = edge_index
        self.edge_label = edge_label
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        return self.edge_index[:, real_idx], self.edge_label[real_idx]


class GraphSAGEEdgeClassifier(nn.Module):
    def __init__(self, num_nodes: int, num_classes: int, hidden_dim: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data, edge_index_for_loss: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index).relu()
            if i < len(self.convs) - 1:  # 在除了最后一层的卷积层后添加dropout
                x = self.dropout(x)
        src, dst = edge_index_for_loss
        edge_feat = torch.cat([x[src], x[dst]], dim=-1)
        logits = self.edge_mlp(edge_feat)
        return logits


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def train_one_epoch(model, data, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for edge_idx, labels in loader:
        edge_idx, labels = edge_idx.to(device), labels.to(device)
        # DataLoader stacks edge tensors to shape (batch, 2); convert to (2, batch)
        if edge_idx.dim() == 2 and edge_idx.size(0) != 2 and edge_idx.size(1) == 2:
            edge_idx = edge_idx.t()
        optimizer.zero_grad()
        logits = model(data, edge_idx)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, data, loader, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for edge_idx, labels in loader:
        edge_idx, labels = edge_idx.to(device), labels.to(device)
        if edge_idx.dim() == 2 and edge_idx.size(0) != 2 and edge_idx.size(1) == 2:
            edge_idx = edge_idx.t()
        logits = model(data, edge_idx)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triangle-file", type=Path, default=Path("/home/hello/glq/code/Album_social_relation/data/train_test_split/triangle_relations_corrected.txt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--add-noise", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(args.seed)

    edges = load_triangle_edges(args.triangle_file, add_noise=args.add_noise)
    data, sup_edge_index, sup_edge_label = build_graph(edges)

    train_idx, val_idx, test_idx = split_indices(len(sup_edge_label), args.train_ratio, args.val_ratio, args.seed)

    train_loader = DataLoader(EdgeDataset(sup_edge_index, sup_edge_label, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EdgeDataset(sup_edge_index, sup_edge_label, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EdgeDataset(sup_edge_index, sup_edge_label, test_idx), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    data = data.to(device)

    num_classes = int(sup_edge_label.max().item() + 1)
    model = GraphSAGEEdgeClassifier(
        num_nodes=data.num_nodes, 
        num_classes=num_classes, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.layers, 
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, data, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, data, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    
    # 保存为best_model2.pth
    torch.save(model.state_dict(), "/home/hello/glq/code/Album_social_relation/classifier/best_model2.pth")
    
    test_loss, test_acc = eval_epoch(model, data, test_loader, device)
    print(f"test_loss {test_loss:.4f} acc {test_acc:.4f} (val_best {best_val:.4f})")


if __name__ == "__main__":
    main()
