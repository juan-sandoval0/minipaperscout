import streamlit as st, json
from scout import fetch_papers, retrieve_evidence, generate_brief

st.title("Mini-PaperScout 📚")
q = st.text_input("Enter a research question", "")
if st.button("Generate brief") and q:
    with st.spinner("Fetching papers…"):
        papers = fetch_papers(q)
        evidence = retrieve_evidence(papers, q)
        brief_md = generate_brief(q, evidence)
    st.markdown(brief_md)