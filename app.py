import streamlit as st
from pdf_loader import load_pdf
from vector_store import embed_and_store
from qa_chain import get_qa_chain, ask_question
from datetime import datetime, date

st.set_page_config(
    page_title="AI PDF QnA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #e6ecf5;
}

.navbar {
    background: linear-gradient(to right, #1f2e63, #35478c);
    padding: 20px;
    color: white;
    font-size: 24px;
    font-weight: 600;
    text-align: center;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    margin-bottom: 30px;
}

.card {
    background-color: #f0f3fa;
    color: #1e1e1e;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    border: 1px solid #c5cddc;
    font-size: 15px;
    line-height: 1.6;
    white-space: pre-wrap;
}

.answer-box {
    background-color: #dde6f7;
    border: 1px solid #b5c8eb;
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
    font-size: 16px;
    color: #12213c;
}

.stButton>button {
    background-color: #35478c;
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 600;
    padding: 8px 16px;
}

.stButton>button:hover {
    background-color: #1f2e63;
    color: #fff;
}

.divider {
    height: 2px;
    background-color: #cbd3e2;
    margin: 40px 0 30px 0;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #1f2e63;
    margin-top: 10px;
}

.footer {
    text-align: center;
    font-size: 15px;
    color: #555;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid #ccc;
}

.footer a {
    color: #35478c;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='navbar'>QUESTIFY = Turn documents into answers</div>", unsafe_allow_html=True)

st.sidebar.title(" About This Project")
st.sidebar.markdown("""
This application allows you to ask questions directly from your PDF documents using the power of AI.

Built with:
- LLM-based question answering
- PDF text extraction
- Embedding search with vector DB

Ideal for:
- Students reading research papers
- Professionals reviewing contracts
- Anyone exploring long documents
""")

st.markdown("<div class='section-title'>Upload your PDF</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    file_path = "uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded successfully.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ask a Question</div>", unsafe_allow_html=True)

    text = load_pdf(file_path)
    vectordb = embed_and_store(text)
    qa = get_qa_chain(vectordb)

    question = st.text_input("Type your question below")

    if question:
        with st.spinner("Processing your question..."):
            answer = ask_question(qa, question)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"<div class='answer-box'><b>Answer ({timestamp}):</b><br><br>{answer}</div>", unsafe_allow_html=True)
else:
    st.info("Please upload a PDF file to begin.")

current_year = date.today().year
footer = f"""
<div class="footer">
    Developed by <strong>Mohammed Ghouse Mohiuddin</strong><br>
    <a href="http://www.linkedin.com/in/md-ghouse-mohiuddin-0622a12a6" target="_blank">LinkedIn</a> | 
    <a href="https://github.com/mohammed-ghouse-99" target="_blank">GitHub</a><br><br>
    &copy; {current_year} All Rights Reserved | Powered by Streamlit and Open AI
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
