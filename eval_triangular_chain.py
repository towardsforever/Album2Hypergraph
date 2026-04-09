import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

# ==================== 知识表示层 ====================

@dataclass
class RelationKnowledge:
    """关系知识基类"""
    relation_id: int
    name: str
    description: str
    properties: Dict[str, Any]
    
class SocialRelationKG:
    """社交关系知识图谱"""
    
    def __init__(self):
        self.relations = {}
        self.rules = []
        self.common_patterns = []
        
    def initialize_relations(self):
        """初始化关系知识"""
        relation_definitions = [
            RelationKnowledge(
                0, "father-child", "父子关系",
                {"directed": True, "transitive": False, "symmetric": False,
                 "hierarchical": True, "family": True, "biological": True,
                 "cardinality": "one-to-many"}
            ),
            RelationKnowledge(
                1, "mother-child", "母子关系",
                {"directed": True, "transitive": False, "symmetric": False,
                 "hierarchical": True, "family": True, "biological": True,
                 "cardinality": "one-to-many"}
            ),
            RelationKnowledge(
                7, "lovers/spouses", "夫妻/恋人关系",
                {"directed": False, "transitive": False, "symmetric": True,
                 "hierarchical": False, "family": True, "biological": False,
                 "cardinality": "one-to-one", "exclusive": True}
            ),
            RelationKnowledge(
                5, "siblings", "兄弟姐妹关系",
                {"directed": False, "transitive": True, "symmetric": True,
                 "hierarchical": False, "family": True, "biological": True,
                 "cardinality": "many-to-many"}
            ),
            RelationKnowledge(
                4, "friends", "朋友关系",
                {"directed": False, "transitive": False, "symmetric": True,
                 "hierarchical": False, "family": False, "biological": False,
                 "cardinality": "many-to-many"}
            ),
            RelationKnowledge(
                15, "colleagues", "同事关系",
                {"directed": False, "transitive": True, "symmetric": True,
                 "hierarchical": False, "family": False, "biological": False,
                 "cardinality": "many-to-many"}
            ),
            RelationKnowledge(
                9, "teacher-student", "师生关系",
                {"directed": True, "transitive": False, "symmetric": False,
                 "hierarchical": True, "family": False, "biological": False,
                 "cardinality": "one-to-many"}
            )
        ]
        
        for rel in relation_definitions:
            self.relations[rel.relation_id] = rel
            
    def add_rule(self, rule_type: str, condition: str, action: str, confidence: float = 1.0):
        """添加规则"""
        self.rules.append({
            "type": rule_type,
            "condition": condition,
            "action": action,
            "confidence": confidence
        })
        
    def initialize_rules(self):
        """初始化基本规则"""
        # 逻辑约束规则
        self.add_rule(
            "logical_constraint",
            "如果A与B是夫妻关系，那么B不能与C也是夫妻关系",
            "夫妻关系具有排他性，一个人不能同时与两人有夫妻关系",
            0.99
        )
        
        self.add_rule(
            "transitive_rule", 
            "如果A是B的父母，B是C的父母，那么A是C的祖父母",
            "亲子关系传递形成祖孙关系",
            0.95
        )
        
        self.add_rule(
            "symmetric_rule",
            "如果A和B是兄弟姐妹，那么B和A也是兄弟姐妹",
            "兄弟姐妹关系是对称的",
            1.0
        )
        
        # 社会常识规则
        self.add_rule(
            "social_norm",
            "师生关系通常不会与夫妻关系同时存在（除非特殊情况）",
            "避免师生恋等不符合社会规范的关系组合",
            0.85
        )
        
        self.add_rule(
            "age_constraint",
            "亲子关系中父母的年龄应大于孩子",
            "考虑年龄约束",
            0.98
        )

# ==================== 大模型智能体 ====================

class SocialRelationAgent:
    """社交关系大模型智能体"""
    
    def __init__(self, 
                 model_name: str = "microsoft/phi-2",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.knowledge_graph = SocialRelationKG()
        self.knowledge_graph.initialize_relations()
        self.knowledge_graph.initialize_rules()
        
        # 初始化大模型（使用较小的模型示例）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 数据存储
        self.training_data = []
        self.pattern_index = None
        self.pattern_embeddings = None
        
        # 缓存
        self.cache = {}
        
    def learn_from_data(self, file_path: str):
        """从数据文件学习模式"""
        print(f"从 {file_path} 学习...")
        
        patterns = []
        pattern_texts = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                    
                a, b, c = map(int, parts[:3])
                r1, r2, r3 = map(int, parts[3:])
                
                # 创建模式描述
                pattern_desc = self._create_pattern_description(r1, r2, r3)
                patterns.append({
                    'nodes': (a, b, c),
                    'relations': (r1, r2, r3),
                    'description': pattern_desc,
                    'line': line_num
                })
                pattern_texts.append(pattern_desc)
                
                self.training_data.append({
                    'r1': r1, 'r2': r2, 'r3': r3,
                    'desc': pattern_desc
                })
                
        # 构建模式索引
        print("构建模式向量索引...")
        embeddings = self.embedding_model.encode(pattern_texts)
        self.pattern_embeddings = np.array(embeddings)
        
        # 使用FAISS进行相似度搜索
        dimension = self.pattern_embeddings.shape[1]
        self.pattern_index = faiss.IndexFlatL2(dimension)
        self.pattern_index.add(self.pattern_embeddings)
        
        print(f"学习了 {len(patterns)} 个模式")
        
    def _create_pattern_description(self, r1: int, r2: int, r3: int) -> str:
        """创建模式的自然语言描述"""
        rel_names = [
            self.knowledge_graph.relations.get(r, RelationKnowledge(r, "unknown", "", {})).name
            for r in [r1, r2, r3]
        ]
        return f"AB是{rel_names[0]}关系，BC是{rel_names[1]}关系，CA是{rel_names[2]}关系"
    
    def retrieve_similar_patterns(self, query: str, k: int = 5) -> List[Dict]:
        """检索相似的模式"""
        if self.pattern_index is None:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.pattern_index.search(np.array(query_embedding), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.training_data):
                data = self.training_data[idx]
                results.append({
                    'relations': (data['r1'], data['r2'], data['r3']),
                    'description': data['desc'],
                    'similarity': 1.0 / (1.0 + dist),
                    'distance': dist
                })
                
        return results
    
    def generate_knowledge_prompt(self, task_description: str) -> str:
        """生成包含知识的提示词"""
        prompt = f"""你是一个社交关系专家，具有以下知识：

关系类型：
{self._format_relation_knowledge()}

约束规则：
{self._format_rules()}

任务：{task_description}

请分析以下社交关系三角形，判断其合理性并给出解释。
"""
        return prompt
    
    def _format_relation_knowledge(self) -> str:
        """格式化关系知识"""
        lines = []
        for rel_id, rel in self.knowledge_graph.relations.items():
            props = []
            for key, value in rel.properties.items():
                if isinstance(value, bool):
                    props.append(f"{key}: {'是' if value else '否'}")
                else:
                    props.append(f"{key}: {value}")
            
            lines.append(f"{rel_id}. {rel.name}: {rel.description}")
            lines.append(f"   属性: {', '.join(props)}")
            
        return "\n".join(lines)
    
    def _format_rules(self) -> str:
        """格式化规则"""
        return "\n".join([
            f"- {rule['condition']} ({rule['type']}, 置信度: {rule['confidence']})"
            for rule in self.knowledge_graph.rules
        ])
    
    def analyze_triangle(self, r1: int, r2: int, r3: int) -> Dict[str, Any]:
        """分析三角形关系"""
        # 步骤1：基于知识图谱的规则检查
        rule_violations = self._check_with_knowledge_graph(r1, r2, r3)
        
        # 步骤2：检索相似模式
        query_desc = self._create_pattern_description(r1, r2, r3)
        similar_patterns = self.retrieve_similar_patterns(query_desc, k=3)
        
        # 步骤3：使用大模型推理
        reasoning_result = self._reason_with_llm(r1, r2, r3, similar_patterns, rule_violations)
        
        # 步骤4：综合评估
        final_score = self._compute_comprehensive_score(
            rule_violations, 
            similar_patterns, 
            reasoning_result
        )
        
        return {
            'relations': (r1, r2, r3),
            'rule_violations': rule_violations,
            'similar_patterns': similar_patterns,
            'reasoning': reasoning_result,
            'comprehensive_score': final_score,
            'recommendation': self._generate_recommendation(
                rule_violations, reasoning_result, final_score
            )
        }
    
    def _check_with_knowledge_graph(self, r1: int, r2: int, r3: int) -> List[str]:
        """使用知识图谱检查规则"""
        violations = []
        
        # 获取关系对象
        rels = [
            self.knowledge_graph.relations.get(r1),
            self.knowledge_graph.relations.get(r2),
            self.knowledge_graph.relations.get(r3)
        ]
        
        # 检查夫妻关系排他性
        spouse_count = sum(1 for rel in rels if rel and rel.relation_id == 7)
        if spouse_count > 1:
            violations.append(f"发现{spouse_count}个夫妻关系，违反排他性原则")
            
        # 检查亲子关系方向
        parent_child_rels = [i for i, rel in enumerate(rels) 
                           if rel and rel.relation_id in [0, 1]]
        if len(parent_child_rels) >= 2:
            violations.append("多个亲子关系需要进一步检查辈分一致性")
            
        # 检查对称性
        for i, rel in enumerate(rels):
            if rel and rel.properties.get('symmetric') and not self._check_symmetry(i, r1, r2, r3):
                violations.append(f"关系{rel.name}可能违反对称性")
                
        return violations
    
    def _check_symmetry(self, edge_idx: int, r1: int, r2: int, r3: int) -> bool:
        """检查对称性"""
        # 简化实现：实际需要更复杂的图结构
        return True
    
    def _reason_with_llm(self, r1: int, r2: int, r3: int, 
                        similar_patterns: List[Dict], 
                        violations: List[str]) -> Dict[str, Any]:
        """使用大模型进行推理"""
        # 创建推理提示
        rel_names = [
            self.knowledge_graph.relations.get(r, RelationKnowledge(r, "unknown", "", {})).name
            for r in [r1, r2, r3]
        ]
        
        prompt = self.generate_knowledge_prompt(
            f"分析社交关系三角形：AB是{rel_names[0]}关系，BC是{rel_names[1]}关系，CA是{rel_names[2]}关系。"
        )
        
        # 添加相似模式信息
        if similar_patterns:
            prompt += "\n相似的历史模式：\n"
            for pattern in similar_patterns:
                prompt += f"- {pattern['description']} (相似度: {pattern['similarity']:.2f})\n"
        
        # 添加违规信息
        if violations:
            prompt += "\n已发现的潜在问题：\n"
            for violation in violations:
                prompt += f"- {violation}\n"
        
        prompt += "\n请分析：\n1. 这个三角形关系是否合理？\n2. 如果不合理，问题在哪里？\n3. 如何修正？\n"
        
        # 使用大模型生成回答（这里简化为模拟）
        # 实际应用中应该调用大模型API
        reasoning_text = self._simulate_llm_response(r1, r2, r3, similar_patterns, violations)
        
        return {
            'prompt': prompt,
            'response': reasoning_text,
            'confidence': self._estimate_confidence(reasoning_text)
        }
    
    def _simulate_llm_response(self, r1: int, r2: int, r3: int, 
                              similar_patterns: List[Dict], 
                              violations: List[str]) -> str:
        """模拟大模型响应（实际应用应替换为真实LLM调用）"""
        responses = {
            (15, 15, 15): "这是一个常见的同事关系三角形。三人互为同事，关系合理。在职场环境中很常见。",
            (1, 7, 0): "这是家庭关系：母亲(A)的孩子(B)与配偶(C)组成夫妻，同时C是A的孩子。这是合理的家庭结构，比如母亲、孩子和孩子的配偶。",
            (1, 5, 1): "问题：A是B的母亲，B和C是兄弟姐妹，C又是A的母亲。这形成了矛盾循环：一个人不能同时是另一个人的母亲和孩子。需要修正。",
            (7, 7, 4): "严重问题：三角形中出现两个夫妻关系。假设B与A和C都是夫妻关系，这违反了夫妻关系的排他性原则。",
            (0, 0, 0): "三个父亲关系：如果A是B的父亲，B是C的父亲，那么C应该是A的孙子，而不是父亲。关系矛盾。",
            (9, 9, 9): "三个师生关系：可能表示A是B的老师，B是C的老师，C是A的老师。这形成了教学循环，虽然可能但不常见。"
        }
        
        key = (r1, r2, r3)
        if key in responses:
            return responses[key]
        
        # 基于相似模式生成响应
        if similar_patterns:
            best_pattern = similar_patterns[0]
            return f"基于相似模式分析：与{best_pattern['description']}类似（相似度{best_pattern['similarity']:.2f}）。"
        
        return "需要进一步分析这个关系组合的合理性。"
    
    def _estimate_confidence(self, reasoning_text: str) -> float:
        """估计推理置信度（简化实现）"""
        confident_keywords = ['合理', '常见', '正常', '符合']
        uncertain_keywords = ['可能', '或许', '不常见', '需要进一步']
        problem_keywords = ['问题', '矛盾', '违反', '不合理']
        
        text_lower = reasoning_text.lower()
        
        if any(kw in text_lower for kw in problem_keywords):
            return 0.3
        elif any(kw in text_lower for kw in confident_keywords):
            return 0.9
        elif any(kw in text_lower for kw in uncertain_keywords):
            return 0.6
        else:
            return 0.7
    
    def _compute_comprehensive_score(self, 
                                   violations: List[str], 
                                   similar_patterns: List[Dict], 
                                   reasoning_result: Dict) -> float:
        """计算综合评分"""
        score = 100.0
        
        # 规则违反惩罚
        score -= len(violations) * 30
        
        # 相似模式加分
        if similar_patterns:
            avg_similarity = np.mean([p['similarity'] for p in similar_patterns])
            score += avg_similarity * 20
        
        # 推理置信度调整
        score *= reasoning_result['confidence']
        
        return max(0, min(100, score))
    
    def _generate_recommendation(self, violations: List[str], 
                               reasoning_result: Dict, 
                               score: float) -> str:
        """生成建议"""
        if score > 80:
            return "关系合理，无需修正。"
        elif score > 50:
            return "关系基本合理，但有优化空间。"
        else:
            if violations:
                return f"发现{len(violations)}个问题，建议修正：{', '.join(violations)}"
            else:
                return "关系存在潜在问题，建议重新检查。"
    
    def suggest_corrections_with_llm(self, r1: int, r2: int, r3: int) -> List[Dict]:
        """使用大模型生成修正建议"""
        analysis = self.analyze_triangle(r1, r2, r3)
        
        if analysis['comprehensive_score'] > 70:
            return [{
                'relations': (r1, r2, r3),
                'score': analysis['comprehensive_score'],
                'reason': '当前关系合理'
            }]
        
        # 生成修正候选
        candidates = []
        relations_to_try = list(self.knowledge_graph.relations.keys())
        
        # 尝试修改每条边
        for edge_idx in range(3):
            current_rels = [r1, r2, r3]
            
            for new_rel in relations_to_try:
                if new_rel == current_rels[edge_idx]:
                    continue
                    
                new_rels = current_rels.copy()
                new_rels[edge_idx] = new_rel
                new_analysis = self.analyze_triangle(*new_rels)
                
                improvement = new_analysis['comprehensive_score'] - analysis['comprehensive_score']
                
                if improvement > 0:
                    candidates.append({
                        'original': (r1, r2, r3),
                        'corrected': tuple(new_rels),
                        'improvement': improvement,
                        'score': new_analysis['comprehensive_score'],
                        'changed_edge': edge_idx,
                        'changed_from': current_rels[edge_idx],
                        'changed_to': new_rel,
                        'reasoning': new_analysis['reasoning']['response'][:100] + "..."
                    })
        
        # 排序并返回
        candidates.sort(key=lambda x: x['improvement'], reverse=True)
        return candidates[:5]
    
    def explain_decision(self, r1: int, r2: int, r3: int) -> str:
        """生成解释性分析"""
        analysis = self.analyze_triangle(r1, r2, r3)
        
        explanation = f"""
# 社交关系三角形分析报告

## 关系描述
- AB关系: {self.knowledge_graph.relations.get(r1, RelationKnowledge(r1, '未知', '', {})).name}
- BC关系: {self.knowledge_graph.relations.get(r2, RelationKnowledge(r2, '未知', '', {})).name}
- CA关系: {self.knowledge_graph.relations.get(r3, RelationKnowledge(r3, '未知', '', {})).name}

## 规则检查
{self._format_violations(analysis['rule_violations'])}

## 历史模式匹配
{self._format_similar_patterns(analysis['similar_patterns'])}

## 智能推理
{analysis['reasoning']['response']}

## 综合评分
得分: {analysis['comprehensive_score']:.1f}/100
建议: {analysis['recommendation']}
"""
        return explanation
    
    def _format_violations(self, violations: List[str]) -> str:
        if not violations:
            return "✓ 未发现规则违反"
        return "⚠ 发现以下问题：\n" + "\n".join(f"- {v}" for v in violations)
    
    def _format_similar_patterns(self, patterns: List[Dict]) -> str:
        if not patterns:
            return "无相似历史模式"
        
        lines = ["相似历史模式："]
        for pattern in patterns:
            lines.append(f"- {pattern['description']} (相似度: {pattern['similarity']:.2f})")
        return "\n".join(lines)

# ==================== 高级功能：关系预测 ====================

class RelationPredictor:
    """关系预测模块"""
    
    def __init__(self, agent: SocialRelationAgent):
        self.agent = agent
        
    def predict_missing_relation(self, r1: int, r2: int, r3: Optional[int] = None) -> List[Tuple[int, float]]:
        """预测缺失的关系"""
        if r3 is None:
            # 预测CA关系
            candidates = []
            for rel_id in self.agent.knowledge_graph.relations.keys():
                analysis = self.agent.analyze_triangle(r1, r2, rel_id)
                candidates.append((rel_id, analysis['comprehensive_score']))
        elif r2 is None:
            # 预测BC关系
            candidates = []
            for rel_id in self.agent.knowledge_graph.relations.keys():
                analysis = self.agent.analyze_triangle(r1, rel_id, r3)
                candidates.append((rel_id, analysis['comprehensive_score']))
        elif r1 is None:
            # 预测AB关系
            candidates = []
            for rel_id in self.agent.knowledge_graph.relations.keys():
                analysis = self.agent.analyze_triangle(rel_id, r2, r3)
                candidates.append((rel_id, analysis['comprehensive_score']))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:3]

# ==================== 使用示例 ====================

def demonstrate_agent():
    """演示智能体功能"""
    print("初始化社交关系智能体...")
    
    # 创建智能体（实际使用时可能需要调整模型）
    agent = SocialRelationAgent(
        model_name="microsoft/phi-2",  # 可以用其他模型
        device="cpu"
    )
    
    # 学习数据
    agent.learn_from_data("triangle_relations.txt")
    
    # 测试分析
    test_cases = [
        (15, 15, 15),  # 三个同事
        (1, 7, 0),     # 母亲-夫妻-父亲
        (1, 5, 1),     # 有问题的组合
        (7, 7, 4),     # 两个夫妻
        (9, 9, 9),     # 三个师生
    ]
    
    print("\n" + "="*80)
    print("智能体分析演示")
    print("="*80)
    
    for r1, r2, r3 in test_cases:
        print(f"\n分析关系组合 ({r1}, {r2}, {r3}):")
        print("-" * 40)
        
        analysis = agent.analyze_triangle(r1, r2, r3)
        
        print(f"综合评分: {analysis['comprehensive_score']:.1f}/100")
        print(f"建议: {analysis['recommendation']}")
        
        if analysis['comprehensive_score'] < 60:
            print("\n修正建议:")
            corrections = agent.suggest_corrections_with_llm(r1, r2, r3)
            for i, corr in enumerate(corrections[:2], 1):
                if 'reason' in corr and corr['reason'] == '当前关系合理':
                    print(f"  建议{i}: 保持原样")
                else:
                    print(f"  建议{i}: {corr['corrected']} (分数: {corr['score']:.1f}, 改进: {corr['improvement']:.1f})")
        
        if analysis['similar_patterns']:
            print(f"\n找到{len(analysis['similar_patterns'])}个相似模式")
            best = analysis['similar_patterns'][0]
            print(f"最相似: {best['description']} (相似度: {best['similarity']:.2f})")
        
        print("\n推理摘要:")
        print(analysis['reasoning']['response'][:200] + "...")
    
    # 生成详细解释报告
    print("\n" + "="*80)
    print("详细解释报告示例")
    print("="*80)
    report = agent.explain_decision(1, 5, 1)
    print(report)
    
    # 演示预测功能
    print("\n" + "="*80)
    print("关系预测示例")
    print("="*80)
    
    predictor = RelationPredictor(agent)
    predictions = predictor.predict_missing_relation(1, 5, None)  # 预测CA关系
    
    print("已知: AB是母子关系, BC是兄弟姐妹关系")
    print("预测CA关系可能性:")
    for rel_id, score in predictions:
        rel_name = agent.knowledge_graph.relations[rel_id].name
        print(f"  {rel_name}: {score:.1f}分")

if __name__ == "__main__":
    demonstrate_agent()