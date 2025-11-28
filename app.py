import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import json

import streamlit as st
import os

# ────────────────────── FIRST TIME SETUP ONLY ──────────────────────
if not os.path.exists("vectorstore/medical_index") or not os.path.exists("vectorstore/compliance_index"):
    st.warning("First time setup — building vector indexes (6–8 minutes one time only)...")
    st.info("Please wait — this runs only once!")

    with st.spinner("Building Medical Index..."):
        os.system("python medical_rag.py")
    with st.spinner("Building Compliance Index (39k chunks)..."):
        os.system("python compliance_rag.py")

    st.success("Setup complete! App is ready!")
    st.balloons()
    st.rerun()
# Silence harmless warnings
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Medical + Policy RAG", layout="wide")
st.title("Medical QA + Contract Compliance Checker")
st.markdown("**Task 1**: Medical RAG | **Task 2**: Compliance Checker (510 CUAD contracts)")

tab1, tab2 = st.tabs(["Task 1: Medical QA", "Task 2: Policy Compliance Checker"])
# TASK 1
with tab1:
    st.header("Medical Transcription RAG System")
    st.markdown("Ask questions based on 500+ real medical records")

    @st.cache_resource
    def load_medical_rag():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("vectorstore/medical_index", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 6})

        pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512, temperature=0.1)
        llm = HuggingFacePipeline(pipeline=pipe)

        template = """You are a medical assistant. Answer using only the context below.
        Be accurate and professional.

        Context:
        {context}

        Question: {question}
        Answer:"""

        prompt = PromptTemplate.from_template(template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain, retriever

    chain, retriever = load_medical_rag()

    query = st.text_input("Ask a medical question:", placeholder="e.g., What are symptoms of asthma?")
    if query:
        with st.spinner("Searching medical records..."):
            answer = chain.invoke(query)
            docs = retriever.invoke(query)

            st.success("Answer")
            st.write(answer)

            st.markdown("### Retrieved Sources")
            for i, doc in enumerate(docs):
                specialty = doc.metadata.get('specialty', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                with st.expander(f"Source {i+1} • {specialty} • {source}"):
                    st.caption(doc.metadata.get('description', 'No description'))
                    st.write(doc.page_content[:800] + "...")

    st.success("Task 1 Complete!")
# TASK 2
with tab2:
    st.header("Contract Policy Compliance Checker")
    st.success("510 CUAD contracts loaded — 39,025 chunks indexed!")

    with open("rules.json", "r") as f:
        rules = json.load(f)

    @st.cache_resource
    def load_compliance_retriever():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("vectorstore/compliance_index", embeddings, allow_dangerous_deserialization=True)
        return db.as_retriever(search_kwargs={"k": 5})

    retriever = load_compliance_retriever()
    pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_length=400, temperature=0.1)
    llm = HuggingFacePipeline(pipeline=pipe)

    if st.button("Run Full Compliance Report (All 15 Rules)", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, rule in enumerate(rules):
            status.text(f"Checking {i+1}/15: {rule['rule'][:60]}...")

            docs = retriever.invoke(rule["rule"])
            context = "\n\n".join([d.page_content[:700] for d in docs])

            prompt_text = f"""
            Rule: {rule['rule']}
            Contract excerpts:
            {context}

            Answer exactly:
            Status: Compliant / Non-Compliant
            Evidence: [short quote or "Not found"]
            Suggestion: [fix or "None"]
            """
            response = llm.invoke(prompt_text)
            status_tag = "PASS" if "Compliant" in response else "FAIL"

            results.append({
                "Rule": rule["rule"],
                "Status": f"**{status_tag}**",
                "Evidence & Suggestion": response.strip()
            })
            progress.progress((i + 1) / len(rules))

        st.subheader("Compliance Report – 15 Rules Checked")
        st.table(results)
        st.success("Full scan completed!")
        st.balloons()
        status.empty()

    st.info("Click the button to generate a full PASS/FAIL compliance report.")
