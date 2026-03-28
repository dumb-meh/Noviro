# Noviro AI Service 🤖🛍️

An AI-powered e-commerce assistant built with FastAPI, LangGraph, OpenAI, ChromaDB, and Redis.

This service provides:
- 💬 Intelligent chat for shopping-related questions
- 🧠 Semantic knowledge search across products, services, consultations, and specialists
- 🌍 Multilingual user handling (auto-detect + translate for vector search)
- ⚡ Conversation continuity with Redis session history

---

## Table of Contents 📚

1. [Project Overview](#project-overview-)
2. [What Has Been Implemented](#what-has-been-implemented-)
3. [Architecture & Flow](#architecture--flow-)
4. [Tech Stack](#tech-stack-)
5. [Project Structure](#project-structure-)
6. [API Endpoints](#api-endpoints-)
7. [Environment Variables](#environment-variables-)
8. [How to Run (Local)](#how-to-run-local-)
9. [How to Run (Docker)](#how-to-run-docker-)
10. [How to Implement / Integrate in Your Product](#how-to-implement--integrate-in-your-product-)
11. [Testing with cURL](#testing-with-curl-)
12. [Troubleshooting](#troubleshooting-)

---

## Project Overview 🎯

Noviro AI Service is an e-commerce-focused backend assistant that:
- accepts user chat messages,
- checks if the message is e-commerce related,
- decides whether to retrieve fresh semantic context,
- queries vector knowledge collections,
- and returns a friendly answer in the user's language.

It is designed for marketplace, product catalog, and service-commerce experiences.

---

## What Has Been Implemented ✅

### 1) AI Chat Pipeline (LangGraph)
- Guardrail classification:
	- Detects query language
	- Detects follow-up intent
	- Detects whether query is e-commerce related
- Smart routing:
	- Non-e-commerce query -> rejection response
	- Follow-up query -> direct response (skip retrieval)
	- New e-commerce query -> semantic retrieval + response generation
- Response generation:
	- Uses conversation history from Redis
	- Uses contextual snippets from vector collections
	- Responds in the detected user language

### 2) Knowledge Base API (CRUD + Semantic Search)
Implemented for 4 entity types:
- Products 🧾
- Services 🛠️
- Consultations 📞
- Specialists 👩‍⚕️👨‍💼

Each supports:
- Add
- Update
- Delete
- Get by ID
- Get all
- Semantic search

### 3) Vector Database Layer (ChromaDB)
- Persistent ChromaDB client
- OpenAI embedding function
- 4 collections created/managed automatically:
	- `products_index`
	- `services_index`
	- `consultations_index`
	- `specialists_index`

### 4) Session Cache Layer (Redis)
- Stores per-user recent conversation turns
- TTL-based session expiry
- Keeps last N messages to reduce cache bloat

### 5) Deployment Readiness
- Dockerfile included
- docker-compose with app + redis
- Health check endpoint

---

## Architecture & Flow 🧭

### High-level flow
1. User sends message to `POST /chat`.
2. Guardrail node classifies language, follow-up, and e-commerce relevance.
3. Router decides:
	 - Reject, or
	 - Retrieve semantic context, or
	 - Generate direct response.
4. Response node builds final answer (using context + recent history).
5. Session history is updated in Redis.
6. Response is returned.

### Core behavior highlights
- If user message is not in English, the query is translated to English for vector search quality.
- Returned answer is generated in user's original language.
- Follow-up queries skip expensive retrieval when possible.

---

## Tech Stack 🧰

- **Backend API:** FastAPI
- **Workflow Engine:** LangGraph
- **LLM + Embeddings:** OpenAI API
- **Vector Store:** ChromaDB (persistent local path)
- **Session Cache:** Redis
- **Containerization:** Docker + Docker Compose

---

## Project Structure 🗂️

```text
main.py                         # FastAPI app bootstrap + middleware + routes
app/core/config.py              # Centralized environment + chatbot config
app/services/chat/
	chatbot_route.py              # /chat endpoint
	chatbot_schema.py             # Request/response models
	chatbot.py                    # LangGraph chatbot pipeline
app/utils/cache_manager.py      # Redis session history manager
app/utils/knowledge/
	knowledge_route.py            # Knowledge CRUD/search endpoints
	knowledge_schema.py           # Knowledge models + response schemas
	product_knowledge.py          # Product manager
	service_knowledge.py          # Service manager
	consultation_knowledge.py     # Consultation manager
	specialist_knowledge.py       # Specialist manager
app/vectordb/manager.py         # ChromaDB setup + collection manager
Dockerfile
docker-compose.yml
requirements.txt
```

---

## API Endpoints 🌐

### Core
- `GET /` -> welcome message
- `GET /health` -> service health status
- `POST /chat` -> AI assistant chat endpoint

### Chat Request/Response

Request:
```json
{
	"message": "I need a dermatologist consultation",
	"user_id": "user_123"
}
```

Response:
```json
{
	"response": "Sure! Here are the best consultation options..."
}
```

### Knowledge Base (prefix: `/knowledge`)

#### Products
- `POST /knowledge/products`
- `PUT /knowledge/products/{product_id}`
- `DELETE /knowledge/products/{product_id}`
- `GET /knowledge/products/{product_id}`
- `GET /knowledge/products?limit=100`
- `GET /knowledge/products/search?query=...&n_results=5&category=...`

#### Services
- `POST /knowledge/services`
- `PUT /knowledge/services/{service_id}`
- `DELETE /knowledge/services/{service_id}`
- `GET /knowledge/services/{service_id}`
- `GET /knowledge/services?limit=100`
- `GET /knowledge/services/search?query=...&n_results=5&category=...`

#### Consultations
- `POST /knowledge/consultations`
- `PUT /knowledge/consultations/{consultation_id}`
- `DELETE /knowledge/consultations/{consultation_id}`
- `GET /knowledge/consultations/{consultation_id}`
- `GET /knowledge/consultations?limit=100`
- `GET /knowledge/consultations/search?query=...&n_results=5&category=...`

#### Specialists
- `POST /knowledge/specialists`
- `PUT /knowledge/specialists/{specialist_id}`
- `DELETE /knowledge/specialists/{specialist_id}`
- `GET /knowledge/specialists/{specialist_id}`
- `GET /knowledge/specialists?limit=100`
- `GET /knowledge/specialists/search?query=...&n_results=5&category=...&min_rating=4`

---

## Environment Variables 🔐

Create a `.env` file (you can copy from `.env.example`):

```env
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0
CACHE_TTL_HOURS=24

# ChromaDB Configuration
CHROMA_DB_PATH=./chroma_db
CHROMA_EMBEDDING_MODEL=text-embedding-3-small

# Application Configuration
MAX_CONVERSATION_HISTORY=15
FOLLOWUP_DETECTION_WINDOW=5
```

Important:
- `OPENAI_API_KEY` is required for both chat and embeddings.
- If Redis is unavailable, app runs but conversation caching is disabled.

---

## How to Run (Local) 💻

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Start Redis
Option A (Docker):
```bash
docker run -d --name noviro-redis -p 6379:6379 redis:7-alpine
```

Option B: use local Redis installation.

### 3) Configure environment
```bash
cp .env.example .env
```
Then set your real `OPENAI_API_KEY` in `.env`.

### 4) Run the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8085 --reload
```

### 5) Open docs
- Swagger UI: `http://localhost:8085/docs`
- ReDoc: `http://localhost:8085/redoc`

---

## How to Run (Docker) 🐳

### 1) Prepare environment
```bash
cp .env.example .env
```
Set `OPENAI_API_KEY`.

### 2) Start services
```bash
docker compose up --build
```

This starts:
- App on port `8085`
- Redis on port `6379`

### 3) Verify health
```bash
curl http://localhost:8085/health
```

---

## How to Implement / Integrate in Your Product 🧩

Use this section if you want to plug Noviro AI into a web/mobile app quickly.

### Step 1: Seed your knowledge base
Before chat becomes useful, add your real business data:
- products
- services
- consultations
- specialists

Use the `/knowledge/*` POST endpoints to insert records.

### Step 2: Connect your frontend chat UI
From frontend, call `POST /chat` with:
- `message`: user question
- `user_id`: stable identifier per customer (important for session continuity)

### Step 3: Keep user_id consistent
Redis cache keys are per user, so use the same `user_id` across conversation turns.

### Step 4: Handle multilingual users
No frontend changes needed for language detection; backend handles detection and response language.

### Step 5: Add production hardening
Recommended for production:
- Add authentication (JWT/API key gateway)
- Add rate limiting
- Restrict CORS origins
- Add structured logging + monitoring
- Add retry and timeout strategy for OpenAI calls

---

## Testing with cURL 🧪

### 1) Add a product
```bash
curl -X POST "http://localhost:8085/knowledge/products" \
	-H "Content-Type: application/json" \
	-d '{
		"product_id": "p-1001",
		"name": "Vitamin C Serum",
		"description": "Brightening serum for daily skincare",
		"price": 29.99,
		"category": "Skincare",
		"subcategory": "Serum",
		"type": "Beauty",
		"stock_quantity": 120,
		"discount": 10,
		"tags": ["vitamin c", "glow", "face"],
		"about": "Dermatologist tested"
	}'
```

### 2) Search products semantically
```bash
curl "http://localhost:8085/knowledge/products/search?query=best%20serum%20for%20brightening&n_results=3"
```

### 3) Chat with assistant
```bash
curl -X POST "http://localhost:8085/chat" \
	-H "Content-Type: application/json" \
	-d '{
		"message": "I need a skincare product for dull skin",
		"user_id": "customer-42"
	}'
```

### 4) Health check
```bash
curl "http://localhost:8085/health"
```

---

## Troubleshooting 🛠️

### `OpenAI API error`
- Check `.env` has valid `OPENAI_API_KEY`.
- Ensure outbound internet is available from container/host.

### Redis connection failed
- Start Redis on expected URL/port.
- Validate `REDIS_URL` and `REDIS_DB` values.
- App still runs, but history/follow-up quality may reduce.

### Empty or poor search results
- Ensure knowledge data has been inserted first.
- Verify `CHROMA_DB_PATH` points to writable persistent folder.
- Try broader query text and higher `n_results`.

### Docker healthcheck fails
- Confirm app is listening on port `8085`.
- Check container logs for startup exceptions.

---

## Notes for Future Improvements 🚀

- Add authentication and tenant isolation
- Add async OpenAI calls for higher throughput
- Add observability (traces, token usage, latency metrics)
- Add evaluation suite for response quality
- Add CI tests for endpoint contracts and chatbot flows

---

Built with FastAPI + LangGraph for scalable AI commerce experiences 💙
