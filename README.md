# Medical + Policy Compliance RAG System  
Two complete RAG systems in one Streamlit app — both fully working.

# Task 1 – Medical Transcription QA
- 500+ real clinical notes (MTSamples)  
- Ask any medical question → get accurate answer + exact source reports  
- Shows specialty, description, and full context

# Task 2 – Contract Policy Compliance Checker
- Full CUAD v1 dataset (510 real legal contracts)  
- 15 practical compliance rules (Governing Law, Non-Compete, Confidentiality, etc.)  
- One click → beautiful PASS/FAIL table with evidence quotes & suggestions

# Why everything runs locally
- No Gemini, no OpenAI, no Hugging Face cloud API  
- All models downloaded once → work 100% offline forever  
- Zero cost, zero internet needed after setup

# How to Run Everything (Step-by-Step)

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Build Task 1
python medical_rag.py

# 3. Build Task 2 
python compliance_rag.py
# → takes ~60–70 minutes first time (39,025 chunks), has progress bar

# 4. Launch the app
streamlit run app.py