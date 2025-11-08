
import streamlit as st
import requests

st.title("Langchain RAG system.")

if st.button("Emed documents"):
    res = requests.post("http://localhost:8000/embed")
    st.success(res.json()["status"])


query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    response = requests.post("http://localhost:8000/query", json={"query": query})
    st.write("Answer:")
    st.write(response.json()["answer"])

