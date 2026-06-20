#  MindCare: Cloud-Native AI Mental Health Assistant

MindCare is a full-stack, RAG-driven (Retrieval-Augmented Generation) AI assistant designed to provide conversational mental health support. Built with a decoupled microservice architecture, it leverages high-speed LPU inference and serverless vector databases to maintain long-term semantic memory of user sessions.

##  Live Demo
* **Frontend:** [mindcare-frontend-teal.vercel.app](mindcare-frontend-teal.vercel.app)
* **Backend API:** [https://mindcare-backend-2cv2.onrender.com](https://mindcare-backend-2cv2.onrender.com)

##  Architecture & Tech Stack

This project was intentionally engineered to be entirely cloud-native and stateless, demonstrating modern decoupling principles and compute-offloading.

**Frontend:**
* **React & Vite:** Hosted on Vercel's edge network for lightning-fast delivery.
* **State Management:** Session-based UUID routing via `localStorage`.

**Backend (Microservice):**
* **FastAPI (Python):** Hosted on Render. Acts as a lightweight traffic router rather than a heavy processing node.
* **LLM Engine:** Utilizing `openai/gpt-oss-120b` via the OpenAI Python client for high-capacity, conversational token generation.
* **Embeddings (Hugging Face API):** Utilized the serverless Inference API (`all-MiniLM-L6-v2`) to offload heavy PyTorch/ML compute requirements, keeping the FastAPI server memory footprint exceptionally low.

**Data Layer:**
* **Vector Database (Pinecone):** Serverless dense vector indexing (384 dimensions) for semantic search and contextual memory retrieval.
* **Document Database (MongoDB Atlas):** Persistent NoSQL storage mapping 1-to-1 with the frontend's JSON state management.

##  Key Engineering Decisions

* **Compute Offloading to Prevent OOM Errors:** By stripping local `sentence-transformers` from the backend and routing embedding generation through Hugging Face's modernized router (`router.huggingface.co`), the backend's RAM requirement dropped from ~800MB to under 100MB, allowing flawless execution on free-tier cloud containers.
* **Stateless RAG Implementation:** The server retains zero local state. On every request, it fetches the user's raw history from MongoDB, pulls semantically relevant past interactions from Pinecone, and injects both into the LLM system prompt to prevent context-window overflow while maintaining long-term conversational memory.

## ⚙️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/kinshuk-coder/mindcare-backend.git](https://github.com/kinshuk-coder/mindcare-backend.git)
cd mindcare-backend
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables**
Create a `.env` file in the root directory and add your keys:
```text
OPENAI_API_KEY=your_openai_compatible_key
MONGODB_URI=your_mongodb_cluster_url
PINECONE_API_KEY=your_pinecone_key
HF_API_KEY=your_huggingface_read_token
```

**4. Run the development server**
```bash
uvicorn main:app --reload
```

## 👨‍💻 Author
* **Kinshuk** - Sole Architect & Developer
