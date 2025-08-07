from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Load lightweight model for fast inference
model_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=model_pipeline)

def get_qa_chain(vectordb):
    """Return QA chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False
    )

def ask_question(qa, question):
    """Ask a question using the QA chain."""
    response = qa.invoke({"query": question})
    return response["result"]
