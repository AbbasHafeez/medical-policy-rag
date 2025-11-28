import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import time
import tqdm 
print("Searching for contracts in your folder...")
txt_files = glob.glob("data/CUAD_v1/full_contract_txt/**/*.txt", recursive=True)
print(f"Found {len(txt_files)} contract files")

if len(txt_files) == 0:
    print("ERROR: No files found!")
    exit()

documents = []
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    contract_name = os.path.basename(file_path)
    doc = Document(page_content=text, metadata={"source": contract_name})
    documents.append(doc)

print(f"Loaded {len(documents)} contracts → Chunking...")
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
print(f"Created {len(chunks)} chunks")
print("Building FAISS index with progress bar... (this may take 4–8 minutes)")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
)
batch_size = 32
vectorstore = None
pbar = tqdm.tqdm(total=len(chunks), desc="Embedding chunks", unit="chunk")

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)
    pbar.update(len(batch))

pbar.close()
print("FAISS index built successfully!")
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore/compliance_index")
print("")
print("INDEX BUILT SUCCESSFULLY!")