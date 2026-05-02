# Agentic LLM Creation and Monitoring
### ISU Student AI Chatbot — Introduction to LLMs | Semester 6 Final Project

> A 24/7 AI-powered chatbot for **Istinye University (ISU)** new students, built to answer specific academic and administrative questions that call-center staff cannot immediately or accurately handle — reducing wait times and call-center load while delivering instant, grounded answers.

---

## Team Roster

| Name | Student ID | Role |
|------|-----------|------|
| **Zekeriya Dulli** | 2309115377 | Lead Developer & System Architect |
| **Obada Abdulhakim Kharaz** | 2309115277 | Project Manager & Demo Lead |
| **Hamdi ALNAQEEB** | 2309116178 | Prompt Engineer & Agent Designer |
| **Fares STOUHI** | 2309115179 | Data & RAG Pipeline Engineer |
| **Azaa Almousli** | 2309115421 | UI/UX Designer & Front-End Developer |
| **Abdulaziz ALYAHYA** | 2309116441 | Risk Management & Safety Monitor |
| **Leen Safi** | 2309116117 | Evaluation & Testing Lead |

---

## What This System Does

New students at Istinye University frequently call the call center with questions that require accurate, up-to-date knowledge: shuttle schedules, residence permit procedures, OIS portal registration steps, Erasmus eligibility, tuition payment deadlines, library locations and hours, graduation requirements, and more. Staff cannot always answer these immediately or with 100% accuracy.

This system acts as a **24/7 AI chatbot** grounded in a real ISU knowledge base (22 documents, 36 indexed chunks) scraped from official ISU sources. It routes each question to the best-suited agent, retrieves the most relevant ISU documents, and generates a grounded, safe response — all without hallucinating unsupported facts.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| LLM Backend | Local model via **LM Studio** | Default: Gemma or any GGUF model running at `localhost:1234` |
| LLM Fallback | OpenAI GPT-4o | Set `OPENAI_API_KEY` in `.env` and clear `LLM_BASE_URL` |
| Agent Framework | LangChain | `ChatOpenAI` with `base_url` override for LM Studio |
| Vector Database | ChromaDB (local persistent) | HNSW cosine-similarity indexing |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` | 384-dim, CPU-only, ~30–80 ms/query |
| UI | Streamlit | `@st.cache_resource` singleton pattern, chat session state |
| Testing | Pytest | `tmp_path` + `monkeypatch.setenv` for full test isolation |
| Language | Python 3.13 | |

---

## Project Structure

```
Agentic_LLM_Creation_and_Monitoring/
├── agents/
│   ├── orchestrator.py        # Multi-agent router & coordinator (Zekeriya)
│   ├── prompt_engineer.py     # PromptTemplate registry & AgentRole enum (Hamdi)
│   └── safety_monitor.py      # 3-layer guardrails & risk scoring (Abdulaziz)
├── rag/
│   ├── vector_db.py           # ChromaDB client & ingestion pipeline (Fares)
│   └── ingest_isu.py          # 22-document ISU knowledge base loader (Fares)
├── eval/
│   ├── pipeline.py            # Intrinsic (perplexity) & extrinsic evaluation (Leen)
│   └── tests/
│       ├── test_safety.py     # Safety unit tests (5 tests)
│       └── test_rag.py        # RAG unit tests (5 tests)
├── ui/
│   └── app.py                 # Streamlit front-end (Azaa)
├── docs/
│   ├── FINAL_REPORT_DRAFT.md  # Full project report
├── .streamlit/
│   └── config.toml            # fileWatcherType=poll (suppresses torchvision warnings)
├── chroma_db/                 # Persisted vector store (auto-created on ingest)
├── .env.example               # Environment variable template
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Edit `.env`. For **LM Studio** (recommended — no API key needed):
```env
LLM_BASE_URL=http://localhost:1234/v1
MODEL_NAME=<your-model-name-from-LM-Studio>
OPENAI_API_KEY=lm-studio
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=knowledge_base
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

For **OpenAI**:
```env
LLM_BASE_URL=
MODEL_NAME=gpt-4o
OPENAI_API_KEY=sk-...
```

### 2. Activate the virtual environment

```powershell
venv\Scripts\Activate.ps1
```

### 3. Populate the ISU knowledge base

Run this once to load all 22 official ISU documents into ChromaDB:

```powershell
python rag/ingest_isu.py
```

Expected output: `Done. Total chunks in database: 36`

### 4. Start LM Studio (if using local model)

- Open LM Studio → load your model → start the local server on port `1234`

### 5. Launch the UI

```powershell
streamlit run ui/app.py
```

### 6. Run the tests

```powershell
pytest eval/tests/ -v
```

All 10 tests should pass.

---

## ISU Knowledge Base Coverage

The knowledge base is populated from real, official ISU sources (istinye.edu.tr, ois.istinye.edu.tr, kutuphane.istinye.edu.tr, sks.istinye.edu.tr, etc.):

| Topic | Source Label |
|-------|-------------|
| University overview, campuses, faculties | `isu_overview` |
| CS / Software Engineering programs, ECTS | `isu_cs_programs` |
| Academic calendar, semester dates | `isu_academic_calendar` |
| Course registration via OIS portal | `isu_course_registration` |
| Student ID card procedures | `isu_student_id` |
| Exam rules and Make-Up policy | `isu_exam_rules` |
| Attendance policy (70% / 80% rule) | `isu_attendance_policy` |
| Transcripts and official documents | `isu_documents_transcripts` |
| Library locations, hours, borrowing rules | `isu_library` |
| Shuttle timetables (Maslak, Şişli, Bahçelievler) | `isu_transportation_shuttle` |
| Residence permit for international students | `isu_international_students_residence_permit` |
| Erasmus & exchange programs | `isu_erasmus_exchange` |
| Double major & minor GPA rules | `isu_double_major_minor` |
| Internship requirements by department | `isu_internship` |
| Health & psychological services | `isu_health_psychological_services` |
| Dining, cafeterias, campus life | `isu_dining_campus_life` |
| IT, WiFi, Blackboard, Office 365 | `isu_it_wifi_systems` |
| Tuition fees, payment methods, IBAN | `isu_tuition_payment` |
| Student clubs & buddy program | `isu_clubs_buddy_program` |
| Graduation requirements & GPA | `isu_graduation_requirements` |
| Admission requirements & placement | `isu_admission_requirements` |
| Contacts directory (emails & phones) | `isu_contacts_directory` |

---

## Key Features

### Multi-Agent Orchestration
Three specialized agents are automatically selected based on query intent:
- **Researcher** — factual Q&A grounded in ISU knowledge base
- **Analyst** — structured multi-factor analysis and comparisons
- **Summarizer** — concise distillation with bullet-point output

### Dual-Level Safety Monitoring
Every user query is scanned **before** the LLM call. Every LLM response is scanned **after**. Three layers:
1. Radicalization keyword detection
2. Implicit bias phrase scanning
3. Hallucination pattern heuristics (regex)

Risk score: `min(1.0, total_flags × 0.15)`. Flagged responses show a 🚫 error in the UI; safe responses show ✅.

### RAG Pipeline
22 ISU documents → fixed-size chunking (200 words / 30-word overlap) → `all-MiniLM-L6-v2` embeddings → ChromaDB HNSW cosine-similarity index → top-4 chunks injected into agent context at query time.

### Live Document Ingestion
Any new document can be added through the Streamlit sidebar without restarting the app. The knowledge base chunk count updates instantly.

### Evaluation Pipeline
- **Intrinsic:** Perplexity from token log-probabilities
- **Extrinsic:** Task success rate, latency, safety pass rate
- JSON reports exported to `eval/report.json`

---

## Architecture Diagram

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  SafetyMonitor (INPUT CHECK)     │  ← Abdulaziz ALYAHYA
│  Blocks harmful queries early    │
└──────────┬───────────────────────┘
           │ safe
           ▼
┌──────────────────────────────────┐
│     MultiAgentOrchestrator       │  ← Zekeriya Dulli
│   Keyword Router → Agent Pool    │
│  ┌─────────────────────────────┐ │
│  │ Researcher│Analyst│Summar.  │ │  ← Hamdi ALNAQEEB (Prompts)
│  └───────────┬─────────────────┘ │
└──────────────┼────────────────────┘
               │ semantic query (n=4)
               ▼
┌──────────────────────────────────┐
│  VectorDBClient (ChromaDB)       │  ← Fares STOUHI
│  22 ISU docs · 36 chunks         │
│  HNSW cosine retrieval           │
└──────────────┬───────────────────┘
               │ context chunks
               ▼
         [LM Studio / GPT-4o]
               │ raw response
               ▼
┌──────────────────────────────────┐
│  SafetyMonitor (OUTPUT CHECK)    │  ← Abdulaziz ALYAHYA
│  3-layer: radical/bias/halluc.   │
└──────────────┬───────────────────┘
               │ {response, safety_report}
               ▼
┌──────────────────────────────────┐
│  Streamlit UI                    │  ← Azaa Almousli
│  Chat · Safety badge · Details   │
└──────────────┬───────────────────┘
               │ metrics
               ▼
┌──────────────────────────────────┐
│  EvaluationRunner                │  ← Leen Safi
│  Perplexity · Task success rate  │
└──────────────────────────────────┘
```

---

## Sample Questions to Try

```
"How do I register for courses on the OIS portal?"
"What is the shuttle schedule from Maslak?"
"What GPA do I need to apply for a double major?"
"What documents do I need for a residence permit?"
"What are the library hours at Vadi Istanbul Campus?"
"How do I connect to ISU WiFi?"
"What happens if I miss more than 30% of my classes?"
"Who do I contact for tuition payment issues?"
```

---
