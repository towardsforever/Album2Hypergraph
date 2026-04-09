import sys
sys.path.append('/home/hello/glq/code/Album_social_relation/')
from Simple_experiment.straight_input_album import read_and_process_txt
from typing import Optional
from Single_graph import GraphConverter, ModelConfig
from cal_similarity import GraphComparator
import os
import glob
from photo_cluster import PhotoCluster
import json

def read_json_file(folder_path):
    import json
    all_data = []
    
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return all_data
    
    # 方法1: 使用glob查找所有.json文件
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    # 方法2: 如果需要查找子文件夹中的JSON文件
    # json_files = glob.glob(os.path.join(folder_path, "**/*.json"), recursive=True)
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
                
        except json.JSONDecodeError as e:
            print(f"解析错误: {file_path} - {e}")
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
    
    return all_data

def generate_album_relation_data_photo_id(processed_data):
    """按相册去重整理所有 photo_id，避免重复生成同一张图。"""
    photo_graph = {}
    album_photos = {
    album_id: [item['photo_id'] for item in processed_data if item['album_id'] == album_id]
    for album_id in set(item['album_id'] for item in processed_data)
    }

    for item in processed_data:
        photo_id = item['photo_id']
        graph_info = item['edges']
        
        photo_graph[photo_id] = graph_info
        
    return album_photos, photo_graph

class AlbumTranferHyperGraph(GraphConverter):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)

    def album_transfer_hyper_graph(self, album_id, process_data):
        
        from photo_cluster import PhotoCluster
        from cal_similarity import GraphComparator
 
        album_photo, photo_data = generate_album_relation_data_photo_id(process_data)
        photo_ids = album_photo[album_id]
        photo_cluster = PhotoCluster(GraphComparator())
        
        print("添加照片数据...")
        for photo_id in photo_ids:
            photo_cluster.add_photo_data(photo_id, photo_data[photo_id])
        
        # 5. 构建相似度矩阵
        photo_cluster.build_similarity_matrix()
        
        # 6. 进行聚类（选择一种方法）
        print("\n开始聚类...")
        
        # 方法A：基于阈值聚类
        final_clusters = photo_cluster.cluster_photos_threshold(similarity_threshold=0.16)
        
        # 方法B：层次聚类（取消注释使用）
        # final_clusters = photo_cluster.cluster_photos_hierarchical(n_clusters=3)
        
        # 7. 打印结果
        photo_cluster.print_clustering_results()
        
        # 8. 获取最终结果（符合要求的格式）
        result = photo_cluster.get_final_clusters()
        
        return result
    
    
    
if __name__ == "__main__":
    datas = read_json_file("/home/hello/glq/code/Album_social_relation/Convert_image_to_Graph/Robust_VLLM")
    album_photos, photo_graph = generate_album_relation_data_photo_id(datas)
    album_id = "72157623458817379"
    album_transfer_hyper_graph = AlbumTranferHyperGraph()

    output_dir = "/home/hello/glq/code/Album_social_relation/Convert_image_to_Graph/Hyper_Graph_Robust"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for album_id in album_photos.keys():
        result = album_transfer_hyper_graph.album_transfer_hyper_graph(album_id, datas)
        graph_json_file = os.path.join(output_dir, f"{album_id}.json")
        
        graph_data = {
            'album_id': album_id,
            'Hyper_Graph': result
        }
        
        with open(graph_json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
   
    
        