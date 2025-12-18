# SalesGPT: AI Fitness Consultant

SalesGPT is a production-grade conversational AI agent designed to act as a Fitness Product Consultant for Peloton. It leverages Large Language Models (LLMs) to engage users, understand their fitness goals, and recommend suitable products.

The system incorporates **Retrieval-Augmented Generation (RAG)** for accurate product information and features an **optional Graph RAG** integration using Neo4j for deeper knowledge retrieval.

## 🚀 Features

- **Context-Aware Conversations**: Maintains conversation history and dynamically determines the stage of the sales process (e.g., Introduction, Needs Analysis, Closing).
- **Vector Search RAG**: Utilizes a FAISS vector store to retrieve semantic matches for product queries.
- **Graph RAG (Optional)**: Integrates with Neo4j to query structured relationships between products and entities, providing richer context when credentials are provided.
- **Fail-Safe Architecture**: Gracefully degrades to standard conversation if tools or databases are unavailable.
- **Modern UI**: Built with Gradio for an accessible chat interface.
- **Dockerized**: Ready for deployment with a production-ready Dockerfile.

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://www.langchain.com/) (Agents, Chains, Tools)
- **LLM**: Azure OpenAI (GPT-4 / GPT-3.5)
- **Vector DB**: FAISS
- **Graph DB**: Neo4j (Optional)
- **Interface**: [Gradio](https://gradio.app/)
- **Runtime**: Python 3.11

## 📂 Project Structure

```
salesGPT/
├── sales_gpt/              # Core application package
│   ├── agents.py           # Main SalesGPT agent logic
│   ├── chains.py           # Conversation & Stage Analyzer chains
│   ├── tools.py            # Tool definitions (Vector & Graph search)
│   ├── graph.py            # Neo4j Graph RAG implementation
│   └── templates.py        # Prompt templates
├── experiments/            # Notebooks and research artifacts
├── app.py                  # Application entry point
├── Dockerfile              # Docker build configuration
├── requirements.txt        # Python dependencies
└── .env                    # Environment configuration (not committed)
```

## ⚡ Getting Started

### Prerequisites

- Python 3.10+
- Azure OpenAI API credentials
- (Optional) Neo4j Database for Graph RAG

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd salesGPT
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   Copy the example environment file and fill in your details:

   ```bash
   cp .env.example .env
   ```

   **Required `.env` variables**:
   - `BASE_URL`: Your Azure OpenAI Endpoint
   - `API_KEY`: Your Azure OpenAI API Key
   - `OPENAI_API_VERSION`: e.g., `2023-05-15`

   **Optional for Graph RAG**:
   - `NEO4J_URI`: e.g., `neo4j+s://<db_id>.databases.neo4j.io`
   - `NEO4J_USERNAME`: `neo4j`
   - `NEO4J_PASSWORD`: Your Neo4j password

### Running the Application

Start the Gradio interface:

```bash
python app.py
```

The application will launch at `http://localhost:7860`.

## 🐳 Docker Deployment

Build and run the containerized application:

```bash
# Build the image
docker build -t sales-gpt .

# Run the container (passing environment variables)
docker run -p 7860:7860 --env-file .env sales-gpt
```

## 🧠 Knowledge Base

The project uses a pre-generated vector store (`docsearch.pkl`) containing product information.

- **Experiments**: Jupyter notebooks used to generate this data and prototype the Graph RAG logic have been moved to the `experiments/` directory for reference.

## 🤝 Contribution

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
