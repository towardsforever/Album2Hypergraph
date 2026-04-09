import networkx as nx
import re
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
from cal_similarity import GraphComparator

class PhotoCluster:
    def __init__(self, graph_comparator: GraphComparator = GraphComparator()):
        """
        照片聚类系统
        
        Args:
            graph_comparator: 图比较器实例
        """
        self.comparator = graph_comparator
        self.photo_data = {}  # 存储photo_id到三元组数据的映射
        self.similarity_matrix = None
        self.clusters = None
        self.photo_ids = []
    
    def add_photo_data(self, photo_id: str, edges: List[List[str]]):
        """
        添加照片数据
        
        Args:
            photo_id: 照片ID
        """
        self.photo_data[photo_id] = edges
    
    def add_batch_photos(self, photo_data_dict):
        """
        批量添加照片数据
        
        Args:
            photo_data_dict: {photo_id: triples_text} 的字典
        """
        self.photo_data.update(photo_data_dict)
    
    def build_similarity_matrix(self) -> np.ndarray:
        """
        构建照片之间的相似度矩阵
        
        Returns:
            np.ndarray: 相似度矩阵
        """
        self.photo_ids = list(self.photo_data.keys())
        n = len(self.photo_ids)
        similarity_matrix = np.zeros((n, n))
        
        print("开始构建相似度矩阵...")
        # 计算所有照片对之间的相似度
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0  # 自身相似度为1
                else:
                    photo1_id = self.photo_ids[i]
                    photo2_id = self.photo_ids[j]
                    
                    edges1 = self.photo_data[photo1_id]
                    edges2 = self.photo_data[photo2_id]

                    graph1 = self.comparator.triples_to_graph(edges1)
                    graph2 = self.comparator.triples_to_graph(edges2)
                    
                    # 使用图比较器计算相似度
                    comparison_result = self.comparator.calculate_iou(graph1, graph2)
                    similarity_matrix[i][j] = comparison_result
                    similarity_matrix[j][i] = comparison_result # 对称矩阵
            
            # 进度显示
            if (i + 1) % 10 == 0:
                print(f"已完成 {i + 1}/{n} 个照片的相似度计算")
        
        self.similarity_matrix = similarity_matrix
        print("相似度矩阵构建完成!")
        return similarity_matrix
    
    def cluster_photos_threshold(self, similarity_threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        基于阈值进行照片聚类
        
        Args:
            similarity_threshold: 相似度阈值，大于该值的照片归为一类
            
        Returns:
            Dict[str, List[str]]: 聚类结果，格式 {'1':[photo1,photo2,...], '2':[photo3,photo4,...]}
        """
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        n = len(self.photo_ids)
        visited = set()
        clusters = {}
        cluster_count = 1  # 从1开始计数
        
        print(f"开始聚类，阈值: {similarity_threshold}")
        
        for i in range(n):
            if i in visited:
                continue
            
            # 创建新类别
            current_cluster = [i]
            visited.add(i)
            
            # 寻找相似的照片
            for j in range(n):
                if j not in visited and self.similarity_matrix[i][j] >= similarity_threshold:
                    current_cluster.append(j)
                    visited.add(j)
            
            # 转换为照片ID，使用字符串键
            photo_cluster = [self.photo_ids[idx] for idx in current_cluster]
            clusters[str(cluster_count)] = photo_cluster
            cluster_count += 1
        
        self.clusters = clusters
        print(f"聚类完成，共 {len(clusters)} 个类别")
        return clusters
    
    def cluster_photos_hierarchical(self, n_clusters: int = 5) -> Dict[str, List[str]]:
        """
        使用层次聚类方法
        
        Args:
            n_clusters: 目标聚类数量
            
        Returns:
            Dict[str, List[str]]: 聚类结果，格式 {'1':[photo1,photo2,...], '2':[photo3,photo4,...]}
        """
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            print("请安装scikit-learn: pip install scikit-learn")
            return {}
        
        print(f"开始层次聚类，目标类别数: {n_clusters}")
        
        # 将相似度转换为距离（1 - 相似度）
        distance_matrix = 1 - self.similarity_matrix
        
        # 层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # 组织聚类结果，使用字符串键
        clusters = defaultdict(list)
        for photo_idx, cluster_label in enumerate(labels):
            photo_id = self.photo_ids[photo_idx]
            clusters[str(cluster_label + 1)] = photo_id  # 从1开始编号
        
        # 转换为普通字典并按类别编号排序
        clusters = {k: clusters[k] for k in sorted(clusters.keys(), key=lambda x: int(x))}
        
        self.clusters = clusters
        print(f"层次聚类完成，共 {len(clusters)} 个类别")
        return clusters
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        获取聚类统计信息
        """
        if self.clusters is None:
            return {}
        
        stats = {
            'total_clusters': len(self.clusters),
            'cluster_sizes': {},
            'avg_cluster_size': 0,
            'max_cluster_size': 0,
            'min_cluster_size': float('inf')
        }
        
        total_photos = 0
        for cluster_id, photos in self.clusters.items():
            size = len(photos)
            stats['cluster_sizes'][cluster_id] = size
            stats['max_cluster_size'] = max(stats['max_cluster_size'], size)
            stats['min_cluster_size'] = min(stats['min_cluster_size'], size)
            total_photos += size
        
        stats['avg_cluster_size'] = total_photos / len(self.clusters) if self.clusters else 0
        stats['total_photos'] = total_photos
        
        return stats
    
    def print_clustering_results(self):
        """打印聚类结果"""
        if self.clusters is None:
            print("尚未进行聚类")
            return
        
        stats = self.get_cluster_statistics()
        
        print(f"\n{'='*50}")
        print(f"照片聚类最终结果")
        print(f"{'='*50}")
        print(f"总照片数: {stats['total_photos']}")
        print(f"聚类数量: {stats['total_clusters']}")
        print(f"平均每类照片数: {stats['avg_cluster_size']:.2f}")
        print(f"最大类大小: {stats['max_cluster_size']}")
        print(f"最小类大小: {stats['min_cluster_size']}")
        
        print(f"\n详细分类:")
        for cluster_id, photos in self.clusters.items():
            print(f"类别 {cluster_id}: {len(photos)}张照片 -> {photos}")
    
    def save_clusters_to_file(self, filename: str = "photo_clusters.json"):
        """保存聚类结果到文件"""
        import json
        
        if self.clusters is None:
            print("没有聚类结果可保存")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.clusters, f, indent=2, ensure_ascii=False)
        
        print(f"聚类结果已保存到: {filename}")
    
    def get_final_clusters(self) -> Dict[str, List[str]]:
        """
        获取最终聚类结果
        
        Returns:
            Dict[str, List[str]]: 格式 {'1':[photo1,photo2,...], '2':[photo3,photo4,...]}
        """
        if self.clusters is None:
            print("请先进行聚类操作")
            return {}
        
        return self.clusters