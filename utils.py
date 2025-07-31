import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Load models only once (globally)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

def get_answers(pdf_url, questions):
    # Download PDF to temp file
    response = requests.get(pdf_url)
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(response.content)
    temp_pdf.close()

    # Load & split PDF
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
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = llm(prompt)[0]["generated_text"]
        answer = response.split("Answer:")[-1].strip()
        answers.append(answer)

    return answers
