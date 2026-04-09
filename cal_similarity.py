import networkx as nx
import re
from typing import Set, Tuple, List
from Single_graph import ModelConfig
from typing import Optional
from openai import OpenAI

class GraphComparator:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.client: Optional[OpenAI] = None
        if not self.config.use_local:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
        self._api_endpoint = self.config.api_base.rstrip('/') + '/chat/completions'

    
    def triples_to_graph(self, edge_list) -> nx.Graph:
        """
        将三元组文本转换为无权图
        
        Returns:
            nx.Graph: 无权图（忽略关系类型）
        """
        G = nx.Graph()

        
        # 添加边（忽略关系类型，只关注实体连接）
        for edges in edge_list:
            e1 = edges[0]
            e2 = edges[1]
            rel = edges[2]
            G.add_edge(e1, e2, relation=rel, original_triple=(e1, rel, e2))  # 无权图，不存储关系类型
        
        return G
    
    def calculate_iou(self, graph1: nx.Graph, graph2: nx.Graph) -> float:
        """
        计算两个图的边交并比
        
        Args:
            graph1: 第一个图
            graph2: 第二个图
        
        Returns:
            float: 边交并比，范围[0, 1]
        """
        def build_edge_relation_map(graph):
            mapping = {}
            for u, v, data in graph.edges(data=True):
                edge_key = tuple(sorted([str(u), str(v)]))
                mapping[edge_key] = data.get('relation', '')
            return mapping
        
        map1 = build_edge_relation_map(graph1)
        map2 = build_edge_relation_map(graph2)
        
        # 获取所有边（基于节点对）
        all_edges = set(map1.keys()) | set(map2.keys())
        
        # 计算特殊交集：节点相同但关系不同的边
        special_intersection = set()
        for edge in all_edges:
            rel1 = map1.get(edge, None)
            rel2 = map2.get(edge, None)
            
            # 两个图都有这条边
            if rel1 is not None and rel2 is not None:
                # 关系不同才计入交集
                if rel1 != rel2:
                    special_intersection.add(edge)
        
        # 计算特殊并集：所有边（不管关系是否一致）
        special_union = all_edges
        
        # 计算特殊IoU
        if len(special_union) == 0:
            special_iou = 0.0
        else:
            special_iou = len(special_intersection) / len(special_union)
        
        # 计算常规指标用于对比
        # 节点IoU
        nodes1 = set(graph1.nodes())
        nodes2 = set(graph2.nodes())
        if nodes1.issubset(nodes2) or nodes2.issubset(nodes1):
            node_iou = 1.0
        else:
            node_intersection = nodes1 & nodes2
            node_union = nodes1 | nodes2
            node_iou = len(node_intersection) / len(node_union) if node_union else 0.0
        
        return 0.5 * special_iou + 0.5 * node_iou


    def compare_graphs_from_triples(self, triples1: str, triples2: str) -> dict:
        """
        从三元组文本直接比较两个图
        
        Args:
            triples1: 第一个图的三元组
            triples2: 第二个图的三元组
        
        Returns:
            dict: 包含比较结果的字典
        """
        # 转换为图
        G1 = self.triples_to_graph(triples1)
        G2 = self.triples_to_graph(triples2)
        
        # 计算边IoU
        edge_iou = self.calculate_iou(G1, G2)
        
        # 获取详细统计信息
        edges1 = set(frozenset([u, v]) for u, v in G1.edges())
        edges2 = set(frozenset([u, v]) for u, v in G2.edges())
        
        intersection = edges1 & edges2
        union = edges1 | edges2
        only_in_G1 = edges1 - edges2
        only_in_G2 = edges2 - edges1
        
        return {
            'edge_iou': edge_iou,
            'graph1_edges': len(edges1),
            'graph2_edges': len(edges2),
            'intersection_edges': len(intersection),
            'union_edges': len(union),
            'only_in_graph1': len(only_in_G1),
            'only_in_graph2': len(only_in_G2),
            'common_edges': [tuple(edge) for edge in intersection],
            'graph1_only_edges': [tuple(edge) for edge in only_in_G1],
            'graph2_only_edges': [tuple(edge) for edge in only_in_G2]
        }
        
    