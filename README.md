# Lab 18: Production RAG Pipeline

**K4-Track3B · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ✅ Có | M2 Dense Search |
| Python 3.11+ | ✅ Có | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `GROQ_API_KEY` | ⚠️ M4+M5 | RAGAS eval (M4), Enrichment LLM (M5) |

**Pre-download models** (tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3B-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3B-Production-RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền GROQ_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền GROQ_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

## Cấu trúc repo

```
K4-Track3B-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3B-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3B-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **11h59 ngày hôm sau diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).

## Groq configuration

This checkout uses Groq for answer generation, M5 enrichment and M4 RAGAS judging.
Groq's OpenAI-compatible endpoint is used with the existing client libraries;
requests go to Groq and do not require an OpenAI API key.
RAGAS answer relevancy uses local bge-m3 embeddings.

Set `GROQ_API_KEY` and optionally `GROQ_MODEL` in `.env`.
The default model is `openai/gpt-oss-120b`; choose another available Groq model if needed.

Windows CMD checks:

```cmd
.venv\Scripts\activate.bat
set HF_HUB_OFFLINE=1
set LAB18_OFFLINE=1
python -m pytest tests/ -q -p no:cacheprovider
set LAB18_OFFLINE=
python check_groq.py
python demo_m345.py
```

`check_groq.py` makes a small real API request and never prints the key.
Offline mode uses fallback enrichment and skips actual RAGAS scoring.
Run `python main.py` for the full baseline and production comparison after the connection succeeds.

## Submission details for this checkout

- Student: BuiQuocViet / 2A202602884.
- GitHub repository: `K4-Track3B-DAY18-BuiQuocViet-2A202602884-ProductionRAG` (public).
- Reflection: `analysis/reflections/reflection_BuiQuocViet.md`.
- Evaluation uses Groq GPT-OSS 120B, local bge-m3 embeddings, answer relevancy strictness 1 and one worked example per RAGAS prompt. Baseline and Production use the same judge configuration.
- API requests wait according to Groq token-limit headers. Evaluation caches each successful question; enrichment caches successful AI results in `.run_cache/`, excluded from Git.
- `check_lab.py` requires a successful full evaluation, completed analysis and reflection, no TODO markers, and 100% unit-test success. Unit tests run offline to avoid API charges.

Enrichment uses `GROQ_ENRICHMENT_MODEL=openai/gpt-oss-20b`; answer generation and RAGAS use GPT-OSS 120B. Model-specific enrichment cache lets preprocessing run alongside baseline evaluation.
