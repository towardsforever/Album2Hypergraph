import torch
import torch.nn as nn
import json
import os
from collections import defaultdict
from pathlib import Path
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

# 关系映射字典
str_to_label = {
    'father-child': 0, 'mother-child': 1, 'grandpa-grandchild': 2, 'grandma-grandchild': 3,
    'friends': 4, 'siblings': 5, 'classmates': 6, 'lovers/spouses': 7,
    'presenter-audience': 8, 'teacher-student': 9, 'trainer-trainee': 10, 'leader-subordinate': 11,
    'band members': 12, 'dance team members': 13, 'sport team members': 14, 'colleagues': 15
}

# 标签映射到字符串
label_to_str = {v: k for k, v in str_to_label.items()}

# 导入训练好的模型结构
class GraphSAGEEdgeClassifier(nn.Module):
    def __init__(self, num_nodes: int, num_classes: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data, edge_index_for_loss: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        src, dst = edge_index_for_loss
        edge_feat = torch.cat([x[src], x[dst]], dim=-1)
        logits = self.edge_mlp(edge_feat)
        return logits


def load_model(model_path):
    """
    加载训练好的GNN模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        model: 加载好的模型
    """
    # 首先加载权重，获取正确的节点数量
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 从state_dict中获取节点数量
    node_emb_weight_shape = state_dict['node_emb.weight'].shape
    num_nodes = node_emb_weight_shape[0]
    hidden_dim = node_emb_weight_shape[1]
    
    # 获取分类器的输出维度，即关系类别数量
    num_classes = state_dict['edge_mlp.2.weight'].shape[0]
    
    # 获取GNN层数
    num_layers = 0
    while f'convs.{num_layers}.lin_l.weight' in state_dict:
        num_layers += 1
    
    # 创建模型
    model = GraphSAGEEdgeClassifier(num_nodes, num_classes, hidden_dim, num_layers)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_local_graph(edges):
    """
    为每个album构建局部图
    
    Args:
        edges: 边列表，格式为 [(u, v, label), ...]
    
    Returns:
        data: PyG的Data对象
        edge_index: 边索引
        edge_labels: 边标签
        id_map: 节点ID映射
    """
    # 获取所有节点
    all_nodes = set()
    for u, v, _ in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    
    # 构建节点ID映射
    id_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    num_nodes = len(id_map)
    
    # 构建边索引和标签
    edge_index_list = []
    edge_labels = []
    
    for u, v, label in edges:
        ui = id_map[u]
        vi = id_map[v]
        edge_index_list.append((ui, vi))
        edge_index_list.append((vi, ui))  # 添加反向边
        edge_labels.append(label)
        edge_labels.append(label)
    
    # 转换为PyTorch张量
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    
    # 创建Data对象
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    
    return data, id_map


def correct_tri_chains(album_id, tri_chains, model, id_map, threshold=0.0005):
    """
    纠正三角链中的边关系
    
    Args:
        album_id: album ID
        tri_chains: 三角链列表
        model: 训练好的模型
        id_map: 节点ID映射
        threshold: 概率阈值，低于该阈值的边需要纠正
    
    Returns:
        corrected_chains: 纠正后的三角链列表
    """
    corrected_chains = []
    
    # 收集所有需要预测的边
    all_edges = set()
    for chain in tri_chains:
        a, b, c, ab_rel, bc_rel, ac_rel, ab_photo, bc_photo, ac_photo = chain
        
        # 确保人物顺序一致
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > c:
            a, c = c, a
        
        all_edges.add((a, b, ab_rel, ab_photo))
        all_edges.add((b, c, bc_rel, bc_photo))
        all_edges.add((a, c, ac_rel, ac_photo))
    
    # 将边转换为模型需要的格式
    edge_list = [(u, v, rel) for u, v, rel, photo in all_edges]
    data, local_id_map = build_local_graph(edge_list)
    
    # 为每条边创建预测用的边索引
    pred_edge_indices = []
    pred_edge_info = []
    
    for u, v, rel, photo in all_edges:
        if u in local_id_map and v in local_id_map:
            ui = local_id_map[u]
            vi = local_id_map[v]
            pred_edge_indices.append((ui, vi))
            pred_edge_info.append((u, v, rel, photo))
    
    # 如果没有需要预测的边，直接返回原三角链
    if not pred_edge_indices:
        return tri_chains
    
    # 转换为张量
    pred_edge_index = torch.tensor(pred_edge_indices, dtype=torch.long).t().contiguous()
    
    # 使用模型进行预测
    with torch.no_grad():
        logits = model(data, pred_edge_index)
        probabilities = torch.softmax(logits, dim=-1)
    
    # 保存预测结果
    edge_predictions = {}
    for i, (u, v, rel, photo) in enumerate(pred_edge_info):
        prob = probabilities[i].tolist()
        max_prob, predicted_label = max(zip(prob, range(len(prob))))
        
        # 检查是否需要纠正
        if prob[rel] < threshold:
            corrected_rel = predicted_label
            print(f"纠正边 ({u}, {v}) 的关系: {label_to_str[rel]} → {label_to_str[corrected_rel]} (概率: {prob[rel]:.4f} → {max_prob:.4f})")
            edge_predictions[(u, v)] = (corrected_rel, max_prob, photo)
        else:
            edge_predictions[(u, v)] = (rel, prob[rel], photo)
    
    # 纠正三角链
    for chain in tri_chains:
        a, b, c, ab_rel, bc_rel, ac_rel, ab_photo, bc_photo, ac_photo = chain
        
        # 确保人物顺序一致
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > c:
            a, c = c, a
        
        # 获取纠正后的关系
        ab_new_rel, ab_prob, ab_new_photo = edge_predictions[(a, b)]
        bc_new_rel, bc_prob, bc_new_photo = edge_predictions[(b, c)]
        ac_new_rel, ac_prob, ac_new_photo = edge_predictions[(a, c)]
        
        # 创建纠正后的三角链
        corrected_chain = (a, b, c, ab_new_rel, bc_new_rel, ac_new_rel, ab_new_photo, bc_new_photo, ac_new_photo)
        corrected_chains.append(corrected_chain)
    
    return corrected_chains


def process_album(album_file, model, output_dir):
    """
    处理单个album的三角链数据
    
    Args:
        album_file: album文件路径
        model: 训练好的模型
        output_dir: 输出目录
    """
    # 读取三角链数据
    with open(album_file, 'r', encoding='utf-8') as f:
        tri_chains_data = json.load(f)
    
    # 转换为内部格式
    tri_chains = []
    for chain in tri_chains_data:
        a = chain['A']
        b = chain['B']
        c = chain['C']
        # 使用str_to_label将字符串关系转换为数字
        ab_rel = str_to_label.get(chain['AB'], -1)  # 如果没有找到，使用-1
        bc_rel = str_to_label.get(chain['BC'], -1)
        ac_rel = str_to_label.get(chain['AC'], -1)
        ab_photo = chain['AB_photo']
        bc_photo = chain['BC_photo']
        ac_photo = chain['AC_photo']
        
        # 确保所有关系都有效
        if ab_rel != -1 and bc_rel != -1 and ac_rel != -1:
            tri_chains.append((a, b, c, ab_rel, bc_rel, ac_rel, ab_photo, bc_photo, ac_photo))
        else:
            print(f"警告：Album {album_file.stem} 中有无效的关系标签: AB={chain['AB']}, BC={chain['BC']}, AC={chain['AC']}")
    
    # 构建节点ID映射
    all_nodes = set()
    for chain in tri_chains:
        all_nodes.add(chain[0])
        all_nodes.add(chain[1])
        all_nodes.add(chain[2])
    id_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    
    # 纠正三角链
    corrected_chains = correct_tri_chains(album_file.stem, tri_chains, model, id_map)
    
    # 转换为输出格式
    output_data = []
    for chain in corrected_chains:
        a, b, c, ab_rel, bc_rel, ac_rel, ab_photo, bc_photo, ac_photo = chain
        output_data.append({
            'A': a,
            'B': b,
            'C': c,
            'AB': ab_rel,
            'BC': bc_rel,
            'AC': ac_rel,
            'AB_photo': ab_photo,
            'BC_photo': bc_photo,
            'AC_photo': ac_photo
        })
    
    # 保存结果
    output_file = os.path.join(output_dir, f"{album_file.stem}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Album {album_file.stem} 处理完成，保存到 {output_file}")


def main():
    """
    主函数
    """
    # 定义文件路径
    album_tri_chains_dir = '/home/hello/glq/code/Album_social_relation/Convert_image_to_Graph/Robust_testing/Tri_chains_robust'
    model_path = '/home/hello/glq/code/Album_social_relation/classifier/best_model.pth'
    output_dir = '/home/hello/glq/code/Album_social_relation/Convert_image_to_Graph/Robust_testing/Corrected_chains_robust'
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = load_model(model_path)
    
    # 处理所有album文件
    print(f"正在处理Album_tri_chains文件夹: {album_tri_chains_dir}")
    
    for filename in os.listdir(album_tri_chains_dir):
        if filename.endswith('.json'):
            album_file = Path(os.path.join(album_tri_chains_dir, filename))
            print(f"\n处理Album: {filename}")
            process_album(album_file, model, output_dir)
    
    print(f"\n所有Album处理完成，结果保存到: {output_dir}")


if __name__ == "__main__":
    main()