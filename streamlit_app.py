import streamlit as st
import requests
import json

# 设置页面配置
st.set_page_config(page_title="RAG 知识库调试助手", layout="wide", page_icon="🤖")

st.title("🤖 RAG 知识库调试助手")
st.caption("连接本地 RAG API，可视化检索结果与生成答案")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 参数配置")
    
    # API 地址配置
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000/api/chat")
    
    # 检索参数
    top_k = st.slider("Top K (检索数量)", min_value=1, max_value=20, value=5)
    
    st.divider()
    st.markdown("### 关于")
    st.markdown("此工具用于调试 RAG 管道的检索质量和生成效果。")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果是助手回复，且包含源文档信息，则渲染出来
        if message.get("sources"):
            with st.expander(f"📚 参考文档 ({len(message['sources'])})"):
                for idx, doc in enumerate(message['sources']):
                    score_info = []
                    if doc.get('is_reranked'):
                        score_info.append(f"Rerank: {doc.get('rerank_score', 0):.4f}")
                    if doc.get('distance') is not None:
                        score_info.append(f"Vector Dist: {doc.get('distance', 0):.4f}")
                    if doc.get('bm25_score') is not None:
                        score_info.append(f"BM25: {doc.get('bm25_score', 0):.4f}")
                    
                    st.markdown(f"**来源 {idx+1}:** `{doc.get('source', 'Unknown')}` (Page {doc.get('page_number', '-')})")
                    st.caption(" | ".join(score_info))
                    st.text_area("上下文窗口内容", doc.get("window"), height=100, key=f"hist_{len(st.session_state.messages)}_{idx}")
                    st.divider()

# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用 API 并显示回复
    with st.chat_message("assistant"):
        with st.spinner("正在检索文档并生成回答..."):
            try:
                payload = {
                    "query": prompt,
                    "top_k": top_k,
                    "filters": {}
                }
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "未生成回答")
                    sources = data.get("source_documents", [])
                    
                    st.markdown(answer)
                    
                    # 实时显示源文档（折叠状态）
                    with st.expander(f"📚 参考文档 ({len(sources)}) - 点击查看详情"):
                        for idx, doc in enumerate(sources):
                            st.markdown(f"**来源 {idx+1}:** `{doc.get('source', 'Unknown')}`")
                            st.text(doc.get("window", "")[:200] + "...") # 只显示前200字符预览
                    
                    # 保存到历史
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                else:
                    st.error(f"API 错误: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"连接失败: {e}")