# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phan Duy Bảo (MSSV 2A202602767)
**Khóa:** K4 - Track 3B
**Ngày hoàn thành:** 2026-10-04

> Ghi chú trung thực: môi trường cloud nơi mình code chặn `huggingface.co` và `api.openai.com`, nên các số đo cần model
> (bge-m3, bge-reranker, RAGAS) **chưa có**. Những ô đánh dấu `⏳` sẽ điền sau khi chạy `python main.py` ở máy có mạng.
> Các con số còn lại (số chunk, kết quả test) là số đo thật.

---

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu bằng regex rồi so cosine giữa 2 câu liền kề (all-MiniLM-L6-v2), < ngưỡng thì cắt chunk mới. Model được cache ở `_get_semantic_model()` để không load lại mỗi lần gọi. Số chunk ở ngưỡng 0.85: ⏳ |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Trên 26 tài liệu có text layer: basic (500 ký tự) = 51 chunk, hierarchical child (256) = 97 chunk, structure-aware = 106 chunk. Child nhỏ hơn → khớp truy vấn chính xác hơn; mỗi child giữ `parent_id` để có thể trả về parent làm context. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Tách theo header `#`–`###`, giữ header trong chunk và lưu `section` vào metadata. Corpus markdown có header rõ nên chunk bám theo từng mục chính sách (kích thước 86–788 ký tự). |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 bắt từ khóa chính xác (số ngày, tên chính sách), dense bắt ngữ nghĩa. RRF chỉ dùng thứ hạng nên không cần chuẩn hóa hai thang điểm khác nhau. Kiểm tra với encoder giả: query "nghỉ phép năm" trả về cả `nghi_phep_nam_v2023.md` lẫn `v2024.md` ở top-3 → rủi ro nhầm phiên bản là có thật, sẽ xét ở failure analysis. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` (`nghỉ_phép`); phải `replace("_", " ")` để token của query và document khớp nhau. Mình cũng `lower()` cả hai phía. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Chấm lại top-20 của hybrid bằng bge-reranker-v2-m3, lấy top-3. Latency: ⏳ ms/query. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Bọc try/except, NaN → 0.0 để không làm hỏng trung bình. Metric thấp nhất: ⏳ |
| Failure analysis | M4 | `failure_analysis()` | Xếp theo điểm trung bình tăng dần, lấy metric thấp nhất của mỗi câu → map sang Diagnostic Tree (faithfulness → hallucination, recall → thiếu chunk, precision → nhiễu, relevancy → prompt). |
| Contextual embeddings | M5 | `_enrich_single_call()` / `contextual_prepend()` | Chọn combined mode: 1 call/chunk trả JSON gồm summary, câu hỏi giả định, context, metadata. Có fallback heuristic khi không có API key nên pipeline không vỡ. Mức giảm retrieval failure: ⏳ |

---

## Phần 2: Khó khăn & Cách giải quyết

- **Lỗi kỹ thuật gặp phải (exact error message):**
  - `httpx.ProxyError: 403 Forbidden` (và `curl: (56) CONNECT tunnel failed, response 403`) khi `SentenceTransformer('all-MiniLM-L6-v2')` tải model từ `huggingface.co`.
  - `huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled` khi đặt `HF_HUB_OFFLINE=1`.
  - Chạy test trước khi cài thư viện: `ModuleNotFoundError: No module named 'qdrant_client'`.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Sandbox cloud có network policy chỉ cho phép package manager, chặn HuggingFace và OpenAI. Mình xác nhận bằng `curl` tới từng host (huggingface.co, hf-mirror.com, api.openai.com đều 000/403).
  - Vì không tải được model, mình kiểm chứng logic bằng encoder/cross-encoder giả (monkeypatch `_SEMANTIC_MODEL`, `_encoder`, `_model`): Qdrant in-memory, BM25, RRF và rerank đều chạy đúng. 28/37 test pass; 9 test còn lại fail đúng và chỉ vì không tải được model.
  - `recreate_collection` đã deprecated ở qdrant-client 1.19 → dùng `collection_exists` + `delete_collection` + `create_collection`.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - 2 PDF scan (BCTC, Nghị định 13) không có text layer nên `load_documents()` bỏ qua; muốn dùng phải OCR. Mình chưa làm phần này.
  - Pipeline gốc chỉ index child chunk và không trả parent như lý thuyết "retrieve child → return parent"; cần đo trước khi đổi vì parent 2048 ký tự có thể làm giảm context precision.

---

## Phần 3: Action Plan cho Project cá nhân

> Cần chỉnh lại theo project thật của mình; phần dưới là kế hoạch dựa trên những gì lab cho thấy.

### Project: Chatbot hỏi đáp chính sách nội bộ (tiếng Việt)

#### 1. Hiện trạng
- **Pipeline hiện tại:** chunk theo đoạn + dense search + LLM trả lời.
- **Known issues:** nhầm giữa tài liệu cũ và mới (v2023/v2024), bỏ sót từ khóa chính xác, chưa có đánh giá định lượng.

#### 2. Kế hoạch áp dụng
1. [ ] **Chunking:** structure-aware cho tài liệu markdown (bám theo mục), hierarchical cho tài liệu dài.
2. [ ] **Search:** hybrid BM25 + dense + RRF; BM25 cần tách từ tiếng Việt.
3. [ ] **Reranking:** bge-reranker-v2-m3, top-20 → top-3; đo latency trước khi đưa lên production.
4. [ ] **Evaluation:** RAGAS với bộ test 20 câu, theo dõi 4 metric sau mỗi thay đổi.
5. [ ] **Enrichment:** combined single-call (1 call/chunk) để giữ chi phí thấp; thêm metadata phiên bản/ngày hiệu lực để lọc tài liệu superseded.

#### 3. Timeline
- Tuần 1: dựng baseline + bộ test + RAGAS.
- Tuần 2: chunking + hybrid search, đo lại.
- Tuần 3: rerank + enrichment, phân tích failure.
- Tuần 4: OCR cho PDF scan, lọc theo phiên bản tài liệu.
