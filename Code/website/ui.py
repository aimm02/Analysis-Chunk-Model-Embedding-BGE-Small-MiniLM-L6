import streamlit as st
import requests

st.set_page_config(page_title="Chatbot Medis Obal", page_icon="🏥")
st.title("🏥 Chatbot Konsultasi Medis (Hybrid RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan gejala Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari data medis..."):
            try:
                res = requests.get(f"http://127.0.0.1:8000/chat?query={prompt}", timeout=300)
                if res.status_code == 200:
                    answer = res.json().get("answer")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Gagal mendapat respon dari Backend.")
            except Exception as e:
                st.error(f"Koneksi terputus. Pastikan Backend (uvicorn) sudah jalan! Error: {e}")