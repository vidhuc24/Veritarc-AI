# 🔍 Veritarc AI - Evidence Validator
## AI-Powered Verification for Audit, Risk & Compliance

Refer to all the rules files from cursor /rules folder

### 🎯 Bootcamp Final Challenge Submission

This project will address the bootcamp challenge requirements through 7 comprehensive tasks, building an end-to-end AI application for audit evidence validation with intelligent document chat capabilities.

---

## 🎬 Demo Video
📹 **[5-Minute Live Demo](https://www.loom.com/share/YOUR_LOOM_LINK_HERE)**
> *Live demonstration of the Veritarc AI application showing evidence upload, validation, document chat, and results*

---

## 📋 Challenge Tasks Plan

### Task 1: Problem & Audience Definition ✅
**Problem Statement:** *Audit teams spend 60-70% of their time manually reviewing control evidence documents, leading to inconsistent evaluations, delayed audit cycles, and increased compliance costs.*

**Target User:** Internal Audit Managers and Senior Auditors at mid-to-large enterprises

**Why This Matters:**
Audit professionals are drowning in evidence review work. A typical SOX audit requires reviewing thousands of documents - access reviews, change logs, financial reconciliations, system configurations. Each document must be evaluated against specific control requirements, but this process is entirely manual, subjective, and time-consuming. Senior auditors spend their expertise on repetitive document review instead of risk analysis and strategic recommendations.

This creates a cascade of problems: audit cycles extend beyond deadlines, junior staff make inconsistent evaluations, audit costs spiral upward, and organizations face regulatory scrutiny for incomplete or delayed compliance reporting.

### Task 2: Proposed Solution ✅
**Solution Vision:**
Veritarc AI will transform evidence review from a manual bottleneck into an intelligent, consistent process. Auditors will upload evidence documents (PDFs, screenshots, logs) and specify the control framework and requirement. The system will instantly analyze the document against comprehensive control criteria, providing a confidence-scored assessment with specific recommendations. Additionally, users can engage in natural language conversations with documents to extract insights and understand compliance details.

**Technology Stack:**
- **LLM**: OpenAI GPT-4 - Superior reasoning for complex control requirement interpretation and document chat
- **Embedding Model**: OpenAI text-embedding-3-small - Proven performance for document similarity matching  
- **Orchestration**: LangChain + LangGraph - Enables multi-agent validation workflow and chat pipeline
- **Vector Database**: Qdrant - Fast similarity search for control requirements matching and document retrieval
- **Monitoring**: LangSmith - Real-time performance tracking and debugging
- **Evaluation**: RAGAS - Automated assessment of faithfulness, relevance, and context precision for chat responses
- **User Interface**: Streamlit - Rapid development for prototype demonstration with dual-mode interface
- **Serving**: FastAPI + Docker - Production-ready API with containerized deployment

**Agentic Reasoning:**
- Evidence Analysis Agent: Will extract key information from uploaded PDF documents
- Compliance Validation Agent: Will match evidence against SOX control requirements
- Risk Assessment Agent: Will evaluate completeness and identify potential gaps
- Recommendation Agent: Will suggest specific improvements or next steps
- Document Chat Agent: Will provide conversational interface for document exploration and Q&A

### Task 3: Data Strategy ✅
**Data Sources:**
1. **SOX Control Requirements Database**: Public SOX framework documentation
2. **Synthetic Evidence Documents**: Will be generated using GPT-4 with known pass/fail labels
3. **Tavily Search API**: Real-time regulatory guidance and best practices
4. **Public SEC Filings**: Real-world SOX compliance documentation examples

**Chunking Strategy:** Semantic chunking with 500-token overlaps to preserve logical relationships in SOX control requirements while ensuring no critical connections are lost during retrieval.

### Task 4: End-to-End Prototype 🔄
Will build complete agentic RAG application with:
- PDF file upload processing
- Multi-agent validation workflow using LangGraph
- Interactive document chat interface for Q&A
- Real-time evidence assessment with confidence scoring
- Dual-mode Streamlit interface (Validation + Chat)
- Local FastAPI deployment

### Task 5: Golden Test Dataset 🔄
Will create comprehensive synthetic evaluation dataset with RAGAS framework:
- 100+ evidence scenarios covering high-quality, partial, poor, and edge cases
- Chat Q&A pairs for document interaction evaluation
- Baseline performance metrics to be established
- Automated evaluation pipeline for both validation and chat capabilities

### Task 6: Advanced Retrieval 🔄
Will implement and test multiple advanced retrieval techniques:
- Hybrid Search: Semantic + keyword matching
- Query Expansion: Automatic compliance term expansion
- Multi-hop Retrieval: Cross-reference following
- Metadata Filtering: Framework-specific filtering
- Conversational Retrieval: Context-aware chat responses

### Task 7: Performance Assessment 🔄
Will conduct comprehensive performance comparison across all retrieval methods with quantified improvements using RAGAS metrics for both validation accuracy and chat response quality.

---

## 🎯 Key Features (Planned)

### Core Functionality
- **PDF Document Processing**: Streamlined PDF upload and text extraction
- **SOX Framework Support**: Focused on Sarbanes-Oxley control requirements
- **Confidence Scoring**: AI-generated confidence levels for each validation
- **Gap Analysis**: Specific recommendations for evidence improvement
- **Audit Trail**: Complete logging for regulatory compliance
- **Interactive Document Chat**: Natural language Q&A with uploaded documents

### Advanced AI Capabilities
- **Multi-Agent Validation**: Specialized agents for different validation aspects
- **Conversational Interface**: Chat with documents using natural language
- **Advanced Retrieval**: Hybrid search, query expansion, multi-hop retrieval
- **Continuous Learning**: Model improvement from user feedback
- **Real-time Processing**: Sub-30-second response times
- **Context-Aware Responses**: Chat maintains conversation history and document context

### User Experience
- **Intuitive Interface**: Simple drag-and-drop PDF upload
- **Dual-Mode Interface**: Switch between Validation and Chat modes
- **Instant Results**: Real-time validation with detailed explanations
- **Interactive Q&A**: Ask questions about document content and compliance
- **Export Capabilities**: PDF reports for audit documentation
- **Responsive Design**: Works on desktop and mobile devices

---

## 📊 Performance Targets

### RAGAS Evaluation Goals
| Metric | Validation Baseline | Chat Baseline | Target with Hybrid Search | Target with Query Expansion | Target with Multi-hop |
|--------|-------------|-------------|----------------|------------------|-------------|
| Faithfulness | 0.85 | 0.80 | 0.88 | 0.89 | 0.91 |
| Answer Relevancy | 0.80 | 0.75 | 0.83 | 0.85 | 0.87 |
| Context Precision | 0.75 | 0.70 | 0.82 | 0.84 | 0.86 |
| Context Recall | 0.70 | 0.65 | 0.76 | 0.78 | 0.82 |

### Business Impact Goals
- **Time Savings**: 70% reduction in evidence review time
- **Consistency**: 95% agreement with expert assessments
- **User Engagement**: 80% of users utilize chat feature for document exploration
- **Cost Reduction**: $200K+ annual savings for mid-size audit teams

---

## 🔮 Future Roadmap

### Phase 2 Enhancements
- **Multi-format Support**: DOCX, images with OCR, CSV files
- **Additional Frameworks**: SOC2, ISO27001 compliance support
- **Fine-tuned Models**: Custom embedding models for audit terminology
- **Multi-modal Processing**: Advanced chart and table analysis
- **Integration APIs**: Connect with major GRC platforms
- **Mobile Application**: Native iOS/Android apps
- **Advanced Chat Features**: Multi-document conversations, chat history, bookmarking

### Scaling Opportunities
- **Enterprise Deployment**: Multi-tenant SaaS platform
- **Industry Specialization**: Healthcare, financial services, manufacturing
- **International Expansion**: Support for global compliance frameworks
- **AI Consulting**: Professional services for custom implementations

---

## 🏆 Bootcamp Learning Integration

This project will demonstrate mastery of key bootcamp concepts:

- **Session 2**: RAG implementation with embeddings and vector databases
- **Session 4**: Production-grade LangChain/LCEL pipelines for validation and chat
- **Session 5-6**: Multi-agent systems with LangGraph
- **Session 7**: Synthetic data generation and LangSmith monitoring
- **Session 8**: Comprehensive evaluation with RAGAS for both validation and chat responses
- **Session 9**: Advanced retrieval techniques and optimization

---

## 👨‍💻 Developer

**Vidhu C** - AI Engineering Bootcamp Participant
- GitHub: [@vidhuc24](https://github.com/vidhuc24)
- Project Repository: [AIE_Bootcamp_VC](https://github.com/vidhuc24/AIE_Bootcamp_VC)

---

## 📄 License

This project is part of the AI Makerspace Engineering Bootcamp certification challenge.

---

---

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have UV installed (recommended) or Python 3.12+
uv --version  # Should show UV version
# OR
python3 --version  # Should show Python 3.12+
```

### Installation

#### Option 1: Using UV (Recommended)
```bash
# Initialize UV project and install dependencies
uv init --no-readme
uv add --requirement requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env
```

#### Option 2: Using Traditional Virtual Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env
```

### 🔍 How to Verify Your Environment is Active

After installation, verify your setup is working correctly:

#### Quick Verification Commands
```bash
# Check Python location (should show your project's .venv path)
which python

# Check virtual environment variable
echo $VIRTUAL_ENV

# Test key package imports
python -c "import openai, langchain, streamlit, fastapi; print('✅ All packages working!')"
```

#### Expected Outputs
```bash
# Virtual environment should be active:
which python
# Expected: /path/to/your/project/.venv/bin/python

echo $VIRTUAL_ENV  
# Expected: /path/to/your/project/.venv

# Packages should import successfully:
python -c "import openai, streamlit; print('✅ Working!')"
# Expected: ✅ Working!
```

#### Troubleshooting
- **Empty parentheses `()` in prompt**: Normal with UV environments - your environment is still active
- **`pip` not found**: With UV, use `uv add package-name` instead of `pip install`
- **Package import errors**: Ensure you're in the activated environment

### Run the Application

#### Using UV (Recommended)
```bash
# Start the backend
uv run python backend/main.py

# In another terminal, start the frontend
uv run streamlit run frontend/streamlit_app.py
```

#### Using Traditional Virtual Environment
```bash
# Ensure environment is activated first
source .venv/bin/activate

# Start the backend
cd backend && python main.py

# In another terminal, start the frontend
cd frontend && streamlit run streamlit_app.py
```

### Access the Application
- **Frontend**: http://localhost:8501 (Validation & Chat Interface)
- **API Documentation**: http://localhost:8000/docs

---

## 📁 Project Structure

```
auditflow-ai/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                      # Environment variables template
├── backend/
│   ├── main.py                       # FastAPI application
│   ├── app/
│   │   ├── evidence_processor.py     # Document processing
│   │   ├── validation_engine.py      # RAG + LLM validation
│   │   ├── retrieval_engine.py       # Advanced retrieval methods
│   │   ├── chat_engine.py            # Document chat interface
│   │   └── vector_store.py           # Qdrant vector database management
│   ├── agents/
│   │   ├── evidence_agent.py         # Evidence analysis agent
│   │   ├── compliance_agent.py       # Compliance validation agent
│   │   ├── recommendation_agent.py   # Recommendation agent
│   │   └── chat_agent.py             # Document chat agent
│   └── models/
│       └── schemas.py                # Pydantic models
├── frontend/
│   ├── streamlit_app.py              # Main Streamlit interface (Validation + Chat)
│   └── src/                          # React components (future)
├── data/
│   ├── control_requirements/         # SOX framework documentation
│   ├── synthetic_evidence/           # Generated test evidence
│   ├── enhanced_evidence/            # RAGAS-enhanced evidence
│   ├── evaluation_datasets/          # RAGAS evaluation datasets
│   └── processed/                    # Processed datasets
├── scripts/
│   ├── generate_synthetic_evidence.py # Synthetic data creation
│   ├── ragas_*.py                    # RAGAS integration scripts
│   └── evaluation_*.py               # Evaluation pipelines
├── notebooks/
│   ├── 01_data_generation.ipynb      # Synthetic data creation
│   ├── 02_rag_pipeline.ipynb         # RAG development
│   ├── 03_evaluation.ipynb           # RAGAS evaluation
│   ├── 04_advanced_retrieval.ipynb   # Advanced techniques testing
│   └── 05_chat_interface.ipynb       # Document chat development
├── tests/
│   ├── test_validation_engine.py     # Unit tests
│   ├── test_chat_engine.py           # Chat functionality tests
│   └── test_agents.py                # Agent tests
└── docs/
    ├── CHALLENGE_RESPONSES.md         # Detailed task responses
    ├── TECHNICAL_ARCHITECTURE.md     # System design
    └── EVALUATION_RESULTS.md         # Performance analysis
```

---