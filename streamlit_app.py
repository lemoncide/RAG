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
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000/api/chat/stream")
    
    # 检索参数
    top_k = st.slider("Top K (检索数量)", min_value=1, max_value=10, value=3)
    
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

# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用 API 并显示回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            payload = {
                "query": prompt,
                "top_k": top_k,
                "filters": {}
            }
            
            # 使用流式接口
            with requests.post(api_url, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            content = chunk.decode("utf-8")
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    # 保存到历史
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"API 错误: {response.status_code} - {response.text}")
                    
        except Exception as e:
            st.error(f"连接失败: {e}")