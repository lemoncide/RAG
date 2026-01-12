# === PDF 文本清洗测试脚本 ===
# 目标: 分析从 PDF 提取的文本质量，并试验不同的清洗策略。

# --- 1. 环境设置 ---
import sys
import os
from pathlib import Path
import re

print("--- 1. 环境设置 ---")
# 将项目根目录添加到 Python 路径中
project_root = str(Path(os.getcwd()).parent) if 'M2/RAG' in os.getcwd() else os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from app.components.reader import PDFReader
    print("成功导入 PDFReader.")
except ImportError as e:
    print(f"导入 PDFReader 失败: {e}")
    print("请确保当前工作目录在 RAG 项目的根目录下。")

print(f"项目根目录设置为: {project_root}")

# --- 2. 定义清洗函数 ---
print("\n--- 2. 定义清洗函数 ---")

def is_potential_garbled(text: str, threshold_non_alnum=0.4, min_avg_word_len=2.5) -> bool:
    """启发式函数，判断一段文本是否可能是乱码。"""
    if not text or text.isspace():
        return True
    
    # 检查非字母、数字、和常见标点符号的字符比例
    alnum_punct_chars = sum(1 for char in text if char.isalnum() or char.isspace() or char in '.,-()[]')
    total_chars = len(text)
    non_alnum_ratio = (total_chars - alnum_punct_chars) / total_chars
    if non_alnum_ratio > threshold_non_alnum:
        return True

    # 检查平均词长
    words = text.split()
    if not words:
        return True
    avg_word_len = sum(len(word) for word in words) / len(words)
    if avg_word_len < min_avg_word_len:
        return True
        
    # 检查是否有超长的非单词字符串 (例如 '**********')
    if re.search(r'[^a-zA-Z0-9\s]{7,}', text):
        return True

    return False

def clean_text(text: str) -> str:
    """对文本进行精细清洗。"""
    # 移除竖线、星号等常见干扰符
    text = re.sub(r'[|*]', ' ', text)
    
    # 移除奇怪的下载水印或页脚
    text = re.sub(r'dy Woy papeo[|]umog.*?ATH', ' ', text, flags=re.IGNORECASE)
    
    # 规范化空白字符：将多个空白符合并为一个空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 尝试修复一些由换行导致的单词断裂 (例如 'kine- matic model')
    text = re.sub(r'-\s+', '', text)
    
    return text

print("清洗函数 'is_potential_garbled' 和 'clean_text' 已定义.")

# --- 3. 执行清洗测试 ---
def run_cleaning_test():
    print("\n--- 3. 执行清洗测试 ---")
    pdf_path = os.path.join(project_root, 'data', 'embodia', 'pdf', 'Journal of Field Robotics - 2017 - Paolillo - Autonomous car driving by a humanoid robot.pdf')

    if not os.path.exists(pdf_path):
        print(f"错误: 找不到 PDF 文件: {pdf_path}")
        return

    print(f"目标 PDF: {pdf_path}")
    reader = PDFReader(input_dir=str(Path(pdf_path).parent))
    structured_docs = reader.read()
    target_doc = None
    for doc in structured_docs:
        if Path(doc['source']).name == Path(pdf_path).name:
            target_doc = doc
            break

    if target_doc:
        print("\n--- 开始逐块分析 ---")
        for i, content in enumerate(target_doc['elements']):
            original_text = content['text']
            if not original_text.strip():
                continue

            print(f"\n--- 文本块 {i} ---")
            if is_potential_garbled(original_text):
                print("[结果: 被过滤]")
                print(f"(原因: 检测为乱码)\n原文: {original_text[:250]}...")
            else:
                cleaned_text = clean_text(original_text)
                print("[结果: 保留并清洗]")
                print(f"原文: {original_text}")
                print(f"清洗后: {cleaned_text}")
    else:
        print("从 PDF 文件中处理文档失败。")

# --- 4. 运行主函数 ---
if __name__ == "__main__":
    run_cleaning_test()