import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load embedding model (lightweight)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load small Q&A model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

def get_answers(pdf_url, questions):
    # Download PDF to temp file
    response = requests.get(pdf_url)
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(response.content)
    temp_pdf.close()

    # Load and split PDF
    loader = PyPDFLoader(temp_pdf.name)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create FAISS vector DB
    db = FAISS.from_documents(chunks, embeddings)

    # Generate answers
    answers = []
    for question in questions:
        relevant_docs = db.similarity_search(question, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        result = llm(prompt)[0]["generated_text"]
        answers.append(result.strip())

    return answers
