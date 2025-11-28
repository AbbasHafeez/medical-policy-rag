
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

print("Loading your local dataset...")
csv_path = r"C:\Users\abbas\OneDrive\Desktop\medical-policy-rag\data\mtsamples.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} records → cleaning...")
df = df.dropna(subset=['transcription']).reset_index(drop=True)
df = df.head(500) 

documents = []
for _, row in df.iterrows():
    documents.append(Document(
        page_content=str(row['transcription'])[:5000],
        metadata={
            "specialty": str(row.get('medical_specialty', 'Unknown')),
            "description": str(row.get('description', ''))[:200],
            "source": str(row.get('sample_name', 'Unknown'))
        }
    ))

print(f"Chunking {len(documents)} documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(documents)

print(f"Creating FAISS index with FREE HuggingFace embeddings... ({len(chunks)} chunks)")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore/medical_index")
print("INDEX CREATED SUCCESSFULLY!")