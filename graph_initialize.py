import json
import pandas as pd
import numpy as np
import networkx as nx
import os
import re
from typing import Set, Tuple, List


def simple_read_jsonl(file_path):
    """
    简单读取JSONL文件
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        解析后的数据列表
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"解析错误: {line}")
                print(f"错误信息: {e}")
    
    return data


def group_by_photo(data):
    """
    按(album_id, photo_id)分组数据
    
    Args:
        data: 解析后的数据列表
    
    Returns:
        按(album_id, photo_id)分组的数据字典
    """
    photo_groups = {}
    
    for item in data:
        key = (item['album_id'], item['photo_id'])
        if key not in photo_groups:
            photo_groups[key] = []
        photo_groups[key].append(item)
    
    return photo_groups


def create_photo_graph(photo_data):
    """
    为单个照片创建图谱
    
    Args:
        photo_data: 照片的关系数据列表
    
    Returns:
        创建好的图谱对象
    """
    G = nx.Graph()
    
    # 添加节点和边
    for edge in photo_data:
        person1 = edge['person1_id']
        person2 = edge['person2_id']
        
        # 获取第一个关系
        if edge['pseudo_labels'] and len(edge['pseudo_labels']) > 0:
            relation = edge['pseudo_labels'][0]['relation']
        else:
            relation = 'unknown'
        
        # 添加节点
        if not G.has_node(person1):
            G.add_node(person1)
        if not G.has_node(person2):
            G.add_node(person2)
        
        # 添加边
        if not G.has_edge(person1, person2):
            G.add_edge(person1, person2, relation=relation)
    
    return G


def save_graphs(photo_groups, output_dir):
    """
    将所有照片图谱保存到指定目录
    
    Args:
        photo_groups: 按照片分组的数据
        output_dir: 输出目录路径
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存每个照片的图谱
    for (album_id, photo_id), data in photo_groups.items():
        # 创建图谱
        G = create_photo_graph(data)
        
        # 也可以保存为JSON格式（可选）
        graph_json_file = os.path.join(output_dir, f"{album_id}_{photo_id}.json")
        graph_data = {
            'album_id': album_id,
            'photo_id': photo_id,
            'nodes': list(G.nodes()),
            'edges': [(u, v, d['relation']) for u, v, d in G.edges(data=True)]
        }
        
        with open(graph_json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 输入文件路径
    input_file = "/home/hello/glq/code/Album_social_relation/data/ml_edges_results_train/ml_edges_results_train.jsonl"
    
    # 输出目录
    output_dir = "/home/hello/glq/code/Album_social_relation/Convert_image_to_Graph/Graph"
    
    # 读取数据
    print(f"正在读取文件: {input_file}")
    data = simple_read_jsonl(input_file)
    print(f"共解析到 {len(data)} 条边")
    
    # 按照片分组
    print("正在按照片分组...")
    photo_groups = group_by_photo(data)
    print(f"共包含 {len(photo_groups)} 张照片")
    
    # 创建并保存图谱
    print(f"正在创建并保存图谱到: {output_dir}")
    save_graphs(photo_groups, output_dir)
    print("图谱创建完成!")
