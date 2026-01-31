from typing import List, Dict, Optional

def reciprocal_rank_fusion(results_lists: List[List[Dict]], k: int = 60, weights: Optional[List[float]] = None) -> List[Dict]:
    """
    使用倒数排名融合 (RRF) 融合多个排序的文档列表。
    
    参数:
        results_lists: 列表的列表，其中每个内部列表包含排序后的文档。
        k: RRF 公式中使用的常数，默认为 60。
        weights: 每个结果列表的可选权重列表。
    
    返回:
        单个重新排序的文档列表。
    """
    rrf_scores = {}
    doc_map = {}
    
    if weights is None:
        weights = [1.0] * len(results_lists)
    
    for i, results in enumerate(results_lists):
        weight = weights[i] if i < len(weights) else 1.0
        
        for rank, doc in enumerate(results):
            # 使用 source 和 text 的组合作为唯一标识符
            # 这是一个简单但有效的方法来去重
            doc_id = (doc["source"], doc["text"])
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                # 创建副本以避免修改原始文档
                doc_map[doc_id] = doc.copy()
                
                # 确保分数字段存在并初始化
                if 'distance' not in doc_map[doc_id] or doc_map[doc_id]['distance'] is None:
                    doc_map[doc_id]['distance'] = 0.0
                if 'bm25_score' not in doc_map[doc_id] or doc_map[doc_id]['bm25_score'] is None:
                    doc_map[doc_id]['bm25_score'] = 0.0
            
            # 根据源列表更新特定分数以保留原始值
            # 列表 0 通常是向量，列表 1 是 BM25
            if i == 0: 
                doc_map[doc_id]['distance'] = doc.get('distance', 0.0)
            elif i == 1:
                doc_map[doc_id]['bm25_score'] = doc.get('bm25_score', 0.0)

            # RRF 公式: weight * (1 / (k + rank + 1))，其中 rank 是从 0 开始的
            rrf_scores[doc_id] += weight * (1 / (k + rank + 1))
    
    # 按组合的 RRF 分数降序对文档进行排序
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)
    
    # 创建最终排序列表
    fused_list = [doc_map[doc_id] for doc_id in sorted_doc_ids]
    
    return fused_list