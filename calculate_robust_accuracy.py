#!/usr/bin/env python3

import os
import json
from collections import defaultdict

# 关系映射字典
str_to_label = {
    'father-child': 0, 'mother-child': 1, 'grandpa-grandchild': 2, 'grandma-grandchild': 3,
    'friends': 4, 'siblings': 5, 'classmates': 6, 'lovers/spouses': 7,
    'presenter-audience': 8, 'teacher-student': 9, 'trainer-trainee': 10, 'leader-subordinate': 11,
    'band members': 12, 'dance team members': 13, 'sport team members': 14, 'colleagues': 15
}


def read_graph_labels(graph_dir):
    """
    读取Graph文件夹中的所有JSON文件，获取关系标签
    
    Args:
        graph_dir: Graph文件夹路径
    
    Returns:
        graph_labels: 字典，键为(album_id, photo_id, person1, person2)，值为关系标签
    """
    graph_labels = {}  # (album_id, photo_id, person1, person2) -> label
    
    # 遍历Graph文件夹中的所有JSON文件
    for filename in os.listdir(graph_dir):
        if not filename.endswith('.json'):
            continue
        
        # 读取文件
        file_path = os.path.join(graph_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取album_id和photo_id
        album_id = data.get('album_id', '')
        photo_id = data.get('photo_id', '')
        
        if not album_id or not photo_id:
            continue
        
        # 处理每条边
        edges = data.get('edges', [])
        for edge in edges:
            if len(edge) < 3:
                continue
            
            person1 = str(edge[0])
            person2 = str(edge[1])
            relation = edge[2]
            
            # 确保person1 < person2，避免重复
            if person1 > person2:
                person1, person2 = person2, person1
            
            # 转换关系为数字标签
            if isinstance(relation, str):
                # 如果是字符串关系，使用str_to_label映射转换
                label = str_to_label.get(relation, -1)
            else:
                # 如果已经是数字，直接使用
                label = int(relation)
            
            # 只保存有效的标签
            if label != -1:
                key = (album_id, photo_id, person1, person2)
                graph_labels[key] = label
    
    return graph_labels


def read_true_labels(txt_file):
    """
    读取new_pairwise_face_train_16.txt文件，获取真实标签
    
    Args:
        txt_file: txt文件路径
    
    Returns:
        true_labels: 字典，键为(album_id, photo_id, person1, person2)，值为真实关系标签
    """
    true_labels = {}  # (album_id, photo_id, person1, person2) -> true_label
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 分割行数据
            parts = line.split()
            if len(parts) < 3:
                continue
            
            # 提取相册ID和照片ID
            album_photo_id = parts[0]
            album_id, photo_id = album_photo_id.split('_')
            
            # 提取最后三列：person1_id, person2_id, true_label
            person1 = parts[-3]
            person2 = parts[-2]
            true_label = int(parts[-1])
            
            # 确保person1 < person2，避免重复
            if person1 > person2:
                person1, person2 = person2, person1
            
            # 保存到字典
            key = (album_id, photo_id, person1, person2)
            true_labels[key] = true_label
    
    return true_labels


def read_predicted_labels(graph_dir):
    """
    读取Robust_VLLM文件夹中的所有文件，获取预测标签
    
    Args:
        graph_dir: Robust_VLLM文件夹路径
    
    Returns:
        predicted_labels: 字典，键为(album_id, photo_id, person1, person2)，值为预测关系标签
    """
    # 读取Graph文件夹中的标签
    predicted_labels = read_graph_labels(graph_dir)
    
    return predicted_labels


def calculate_accuracy(true_labels, predicted_labels):
    """
    计算准确率、F1-score等指标
    
    Args:
        true_labels: 真实标签字典
        predicted_labels: 预测标签字典
    
    Returns:
        accuracy: 准确率
        correct: 正确预测的数量
        total: 总预测数量
        class_accuracy: 每种关系的准确率统计
        class_metrics: 每种关系的F1-score等指标
        weighted_f1: 总体加权F1-score
    """
    correct = 0
    total = 0
    
    # 初始化每种关系的统计字典
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    # 初始化TP, FP, FN用于计算F1-score
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)
    
    # 获取所有键（真实标签和预测标签的并集）
    all_keys = set(true_labels.keys()) | set(predicted_labels.keys())
    
    # 遍历所有键
    for key in all_keys:
        if key in true_labels and key in predicted_labels:
            true_label = true_labels[key]
            pred_label = predicted_labels[key]
            if true_label == pred_label:
                correct += 1
                class_correct[true_label] += 1
                TP[true_label] += 1
            else:
                FP[pred_label] += 1
                FN[true_label] += 1
            total += 1
            class_total[true_label] += 1
        elif key in predicted_labels:
            # 预测了但不在真实标签中，视为FP
            pred_label = predicted_labels[key]
            FP[pred_label] += 1
        elif key in true_labels:
            # 真实标签存在但未预测，视为FN
            true_label = true_labels[key]
            FN[true_label] += 1
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 计算每种关系的准确率
    class_accuracy = {}
    for label in range(16):  # 16种关系
        if class_total[label] > 0:
            class_accuracy[label] = {
                'correct': class_correct[label],
                'total': class_total[label],
                'accuracy': class_correct[label] / class_total[label]
            }
        else:
            class_accuracy[label] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.0
            }
    
    # 计算每种关系的F1-score
    class_metrics = {}
    for label in range(16):
        tp = TP[label]
        fp = FP[label]
        fn = FN[label]
        
        # 计算精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # 计算召回率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # 计算F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[label] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # 计算总体加权F1-score（按真实样本数加权）
    total_support = sum(TP[label] + FN[label] for label in range(16))
    weighted_f1 = 0.0
    if total_support > 0:
        for label in range(16):
            support = TP[label] + FN[label]
            weighted_f1 += class_metrics[label]['f1'] * support
        weighted_f1 /= total_support
    
    return accuracy, correct, total, class_accuracy, class_metrics, weighted_f1


def main():
    """
    主函数
    """
    # 定义文件路径
    txt_file = '/home/hello/glq/code/Album_social_relation/data/train_test_split/new_pairwise_face_train_16.txt'
    graph_dir = '/home/hello/glq/code/Album_social_relation/Convert_image_to_Graph/Robust_testing/Robust_VLLM'
    
    # 读取真实标签
    print(f"正在读取真实标签文件: {txt_file}")
    true_labels = read_true_labels(txt_file)
    print(f"成功读取 {len(true_labels)} 个真实关系标签")
    
    # 读取预测标签
    print(f"\n正在读取Robust_VLLM文件夹: {graph_dir}")
    predicted_labels = read_predicted_labels(graph_dir)
    print(f"成功读取 {len(predicted_labels)} 个预测关系标签")
    
    # 计算准确率和F1-score
    print(f"\n正在计算准确率和F1-score...")
    accuracy, correct, total, class_accuracy, class_metrics, weighted_f1 = calculate_accuracy(true_labels, predicted_labels)
    
    # 输出结果
    print(f"\n准确率计算结果:")
    print(f"总预测数量: {total}")
    print(f"正确预测数量: {correct}")
    print(f"准确率: {accuracy:.4f} ({correct}/{total})")
    print(f"总体加权F1-score: {weighted_f1:.4f}")
    
    # 输出每种关系的准确率
    print(f"\n每种关系的准确率统计:")
    print("-" * 80)
    print(f"{'关系':<30} {'正确数':<10} {'总数':<10} {'准确率':<10}")
    print("-" * 80)
    
    # 标签到关系的映射
    label_to_str = {v: k for k, v in str_to_label.items()}
    
    # 打印每种关系的准确率
    for label in range(16):
        rel_name = label_to_str.get(label, f"未知关系 {label}")
        stats = class_accuracy[label]
        print(f"{rel_name:<30} {stats['correct']:<10} {stats['total']:<10} {stats['accuracy']:.4f}")
    print("-" * 80)
    
    # 输出每种关系的F1-score
    print(f"\n每种关系的F1-score统计:")
    print("-" * 120)
    print(f"{'关系':<30} {'TP':<8} {'FP':<8} {'FN':<8} {'精确率':<10} {'召回率':<10} {'F1-score':<10}")
    print("-" * 120)
    
    # 打印每种关系的F1-score
    for label in range(16):
        rel_name = label_to_str.get(label, f"未知关系 {label}")
        metrics = class_metrics[label]
        print(f"{rel_name:<30} {metrics['TP']:<8} {metrics['FP']:<8} {metrics['FN']:<8} "
              f"{metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1']:.4f}")
    print("-" * 120)
    
    # 计算覆盖度：预测标签在真实标签中的覆盖率
    coverage = total / len(true_labels) if len(true_labels) > 0 else 0.0
    print(f"\n覆盖度计算结果:")
    print(f"真实标签总数: {len(true_labels)}")
    print(f"覆盖的真实标签数量: {total}")
    print(f"覆盖度: {coverage:.4f} ({total}/{len(true_labels)})")


if __name__ == "__main__":
    main()
