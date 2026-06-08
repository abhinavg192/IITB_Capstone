# Posly — AI Social Media Content Generator
## Project Context File
## For Abhinav Gupta | Start new chat with: "Hi Claude, I am Abhinav. I want to continue building Posly. Here is the full context file." Then upload this file.

---

## WHAT WAS BUILT (IITB Capstone — Completed May 2026)

A working AI social media content generator demonstrated at IIT Bombay e-PG Diploma in AI & DS capstone presentation. Presentation was successful.

**What it does:**
User enters brand name + topic + platform + tone → uploads brand guidelines PDF → system generates 5 post variants → hybrid scorer ranks them → top variant recommended → analytics shown.

**GitHub:** https://github.com/abhinavg192/IITB_Capstone

---

## TEAM (Capstone only — not ongoing)

| Member | Role | Key Deliverable |
|---|---|---|
| Abhinav Gupta | LangChain Pipeline + Optimizer | modules/pipeline.py |
| Shreyas Shanbhag | RAG + LLM-as-Judge | modules/rag.py, modules/judge.py |
| Khushee Paprunia | DistilBERT Content Scorer | models/distilbert_engagement/ |
| Manas Sandeep Rane | EDA + XGBoost | models/xgboost_model.pkl |
| Aditya Sikka | Streamlit UI | app/app.py |

---

## CURRENT TECH STACK

| Component | Technology | Status |
|---|---|---|
| LLM | Claude Haiku (dev) / Sonnet (prod) via LangChain | ✅ Working |
| Prompt Engineering | 3-level: Zero-shot → Few-shot → COT+JSON | ✅ Working |
| Brand RAG | FAISS + sentence-transformers all-MiniLM-L6-v2 | ✅ Working |
| Engagement Scorer | XGBoost trained on 100K Kaggle Twitter posts | ✅ Working |
| Content Quality | DistilBERT fine-tuned on 50K posts (HuggingFace: abhinavg192/distilbert-engagement) | ✅ Working |
| LLM Judge | Claude-based faithfulness + relevance scorer | ✅ Working |
| Hybrid Score | 0.25×XGBoost + 0.25×DistilBERT + 0.25×Faithfulness + 0.25×Relevance | ✅ Working |
| UI | Streamlit 3-screen app | ✅ Working |
| Platforms | Twitter, LinkedIn, Instagram | ✅ Working |

---

## REPO STRUCTURE

```
IITB_Capstone/
├── .env                          ← API keys (never push)
├── app/
│   └── app.py                    ← Streamlit UI (integrated)
├── data/
│   ├── raw/
│   │   └── Twitterdatainsheets.csv
│   └── brand_guidelines/
│       ├── Adobe_Brand_Voice.pdf
│       ├── Duolingo_Brand_Voice.pdf
│       ├── Adidas_Brand_Voice.pdf
│       ├── Starbucks_Brand_Book.pdf
│       └── Amazon Global_Brand Guidelines.pdf
├── models/
│   ├── xgboost_model.pkl         ← Manas's model (276KB, in repo)
│   ├── xgboost_columns.pkl
│   └── distilbert_engagement/    ← gitignored, hosted on HuggingFace
├── modules/
│   ├── pipeline.py               ← Core: generate_posts(), optimize_variants()
│   ├── predictor.py              ← predict_engagement(), score_content_quality()
│   ├── rag.py                    ← build_index(), retrieve_brand_context()
│   └── judge.py                  ← judge_variants(), LLM-as-Judge
└── notebooks/
    ├── abhinav/
    │   ├── prompt_pipeline.ipynb       ← MAIN development notebook
    │   ├── rag_evaluation.ipynb        ← RAG with vs without comparison
    │   └── prompt_evaluation.ipynb     ← 3-template evaluation framework
    ├── manas/                          ← EDA + XGBoost training
    ├── khushee/                        ← DistilBERT training
    └── shreyas/                        ← RAG pipeline development
```

---

## KEY FUNCTIONS

### modules/pipeline.py
```python
generate_posts(brand_name, topic, tone, platform, pdf_path=None)
# → list of 5 variant dicts with post_text, hashtags, reasoning, suggested_posting_time

optimize_variants(variants, platform)
# → (ranked_variants, scores, scoring_failed)

rerank_with_judge(variants)
# → hybrid reranked list using all 4 scoring components
```

### modules/predictor.py
```python
predict_engagement(features_dict)
# features: caption_length, hashtag_count, new_sentiment_score (VADER),
#           has_cta, platform_encoded (0=Twitter,1=LinkedIn,2=Instagram), hour_posted
# → raw XGBoost score (higher = better)

score_content_quality(text)
# → DistilBERT score 0-1
```

### modules/rag.py
```python
build_index(pdf_path, brand_name)      # chunk → embed → FAISS index
retrieve_brand_context(brand_name, topic, platform, tone)  # → top-3 chunks string
is_index_built(brand_name)
clear_index(brand_name)
```

---

## EVALUATION RESULTS (from prompt_evaluation.ipynb)

| Metric | Zero-shot | Few-shot | Unified |
|---|---|---|---|
| DistilBERT Score | 0.032 | 0.030 | 0.093 (+191%) |
| Judge Faithfulness /10 | 7.1 | 7.3 | 8.0 |
| Judge Relevance /10 | 8.8 | 9.1 | 8.9 |

RAG evaluation (Adobe LinkedIn):
- Without RAG: 13/25 brand alignment score
- With RAG: 21/25 brand alignment score (+62% improvement)
- Key evidence: "stand out and succeed" traced directly to retrieved PDF chunk

XGBoost model:
- Dataset: 100K+ Kaggle Twitter posts
- RMSE: ~0.64 on chronological validation set
- Top feature: hour_posted (0.417 importance)
- Split: 80/20 chronological

DistilBERT model:
- Training: 50K records (Twitter + LinkedIn + Instagram)
- Architecture: DistilBERT base → Linear 768→256 → Linear 256→1 → Sigmoid
- Hosted: huggingface.co/abhinavg192/distilbert-engagement
- 97% of BERT performance at 40% less compute

---

## LOCAL SETUP

```
IDE: Cursor
Python: 3.11.15
Virtual env: /Users/abhinav/Documents/GitHub/.venv
Project: /Users/abhinav/Documents/GitHub/IITB_Capstone
Activate: source /Users/abhinav/Documents/GitHub/.venv/bin/activate
Run app: streamlit run /Users/abhinav/Documents/GitHub/IITB_Capstone/app/app.py
```

**Critical env fix (must be first in every notebook/script):**
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
```

**API keys in .env:**
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
HUGGINGFACE_TOKEN=...
```

---

## KNOWN LIMITATIONS (to fix in production)

1. **FAISS is in-memory** — index lost on restart, not scalable beyond single user
2. **Streamlit is single-threaded** — blocks on concurrent requests
3. **Synchronous Claude calls** — no streaming, no batching
4. **No persistence** — session state only, no database
5. **No auth** — anyone can use and run up API costs
6. **XGBoost trained on Twitter only** — LinkedIn/Instagram scoring is approximate
7. **Hybrid score bug** — rerank_with_judge() normalisation needs fixing (scores showing 0.000)
8. **Trending topics not integrated** — Reddit PRAW module was descoped

---

## PRODUCT VISION (Posly)

**Core differentiator:** Pre-posting engagement prediction — no competitor does this.

**Competitors and gaps:**
- Jasper.ai — good copy, no engagement prediction
- Copy.ai — fast templates, no brand voice RAG
- Buffer AI — scheduling + generation, no scoring
- Hootsuite OwlyWriter — platform-aware, no pre-posting intelligence

**Potential product name:** Posly

**Future integrations:**
- Month 1-2: FastAPI backend, PostgreSQL, Docker
- Month 3-4: Celery queuing, Pinecone replacing FAISS, Redis caching
- Month 4-6: Buffer/Hootsuite API integration
- Month 7-12: HubSpot connector, multi-tenant brand management
- Year 2: Marketo/Salesforce, enterprise tier

**Monetisation angles:**
- SaaS subscription (per brand / per platform / per post)
- API access for agencies
- White-label for marketing platforms
- Enterprise: custom model fine-tuned on client's historical engagement data

---

## SCALABILITY GAPS TO ADDRESS

| Gap | Current | Production Fix |
|---|---|---|
| Vector store | FAISS in-memory | Pinecone / ChromaDB persistent |
| Backend | Streamlit single-thread | FastAPI async + Celery |
| LLM calls | Synchronous | Async + streaming |
| Model serving | pkl loaded per request | BentoML / SageMaker endpoint |
| Database | None (session state) | PostgreSQL + Redis |
| Auth | None | Auth0 / Firebase |

---

## WHAT TO BUILD NEXT

Priority order based on what blocks monetisation:

1. **Fix hybrid scorer bug** — scores showing 0.000 due to normalisation issue in rerank_with_judge()
2. **FastAPI backend** — replace Streamlit logic, enable concurrent users
3. **Persistent FAISS / Pinecone** — pre-index brands, instant retrieval
4. **User auth + billing** — Stripe integration, per-post or subscription model
5. **Streaming responses** — users see text generating in real time
6. **LinkedIn dataset** — retrain XGBoost on LinkedIn-specific engagement data
7. **Trending topics** — complete the Reddit PRAW integration
8. **Image generation** — Claude prompt → DALL-E image per post variant

---

*Generated: May 2026 | IIT Bombay e-PG Diploma in AI & DS Capstone*
*Next session: Continue from scalability fixes and production architecture*
