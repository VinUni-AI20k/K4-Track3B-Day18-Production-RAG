# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phan Duy Bảo (MSSV 2A202602767)
**Khóa:** K4 - Track 3B
**Ngày hoàn thành:** 2026-10-05

> Nguồn số liệu: lần chạy `python main.py` ngày 2026-10-05 (`reports/ragas_report.json`, `naive_baseline_report.json`, `latency_report.json`), 37/37 test pass, và các phép đo chunk chạy trực tiếp trên corpus. Ở bản nháp đầu mình ghi nhầm số chunk (51/97/106) và để `⏳` vì khi đó môi trường cloud chặn HuggingFace/OpenAI; các số dưới đây đã đo lại. Phân tích chi tiết từng failure nằm ở `analysis/failure_analysis.md`.

---

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu bằng regex, so cosine giữa hai câu liền kề (all-MiniLM-L6-v2), nhỏ hơn ngưỡng thì cắt. Trên 26 tài liệu: **ngưỡng 0.85 → 208 chunk** (basic = 57); hạ ngưỡng 0.7 → 182; 0.5 → 43. Ngưỡng 0.85 tạo nhiều chunk hơn cả hierarchical child (103). Mình đoán do MiniLM huấn luyện chủ yếu tiếng Anh nên độ tương đồng giữa các câu tiếng Việt thấp, nhưng chưa kiểm chứng. Pipeline cuối **không dùng** semantic. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | 26 parent, 103 child (child 256 ký tự). Corpus nhỏ nên **mỗi tài liệu chỉ là 1 parent**. Pipeline chỉ index child và trả nguyên child; thí nghiệm trả parent (single run) cho Faithfulness 0.831 so với 0.595 khi trả child (cùng temperature 0, xem failure_analysis mục 4). Đây là bài học lớn nhất của lab. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Tách theo header `#`–`###`, giữ header trong chunk, lưu `section` vào metadata → 107 chunk. Probe BM25 offline: fact-coverage@3 là 0.685, thấp hơn basic (0.742). Chunk nhỏ hơn không tự động tốt hơn. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 bắt từ khóa chính xác, dense bắt ngữ nghĩa, RRF chỉ dùng thứ hạng nên không cần chuẩn hóa hai thang điểm. Hybrid search ≈ 97 ms/query. Hạn chế thấy rõ: với câu "Bao lâu phải đổi mật khẩu", top-20 chứa cả `mat_khau_v1` lẫn `mat_khau_v2` — RRF không phân biệt phiên bản vì cả hai đều khớp từ khóa và ngữ nghĩa. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` (`nghỉ_phép`); phải `replace("_", " ")` và `lower()` ở cả query lẫn document thì token mới khớp. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 chấm lại top-20 → top-3. **Latency trung bình 1168 ms/query**, đã gồm thời gian nạp model ở query đầu (model lazy-load; bước "load reranker" báo 0.0 s), nên latency thuần thấp hơn nhưng mình chưa đo tách. Rerank tách chunk liên quan khỏi nhiễu rất rõ (0.78 so với ~0.05 trong câu laptop) nhưng **chấm v1 và v2 bằng nhau (0.966)**, nên không giải quyết được nhầm phiên bản. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Production: faithfulness 0.708, answer_relevancy 0.688, context_precision 0.946, context_recall 0.883. **Thấp nhất là Answer Relevancy** (baseline cũng thấp nhất ở metric này: 0.726). Nguyên nhân: RAGAS 0.1.22 tính `score = mean_cosine × int(not noncommittal)`, nên chỉ cần một câu trả lời có ý "không tìm thấy" là điểm về 0 — kể cả câu Senior trả đúng 18 ngày phép nhưng thiếu phần lương. Production **không vượt baseline** (chỉ precision tăng +0.02). |
| Failure analysis | M4 | `failure_analysis()` | Xếp theo điểm trung bình tăng dần, lấy metric thấp nhất mỗi câu → Diagnostic Tree. Phát hiện: cây chẩn đoán **chẩn sai** câu laptop. Câu này có faithfulness 0.0 nên bị gán "LLM hallucinating", nhưng nguyên nhân thật là retrieval thiếu chunk bảng phê duyệt; câu trả lời `"Không tìm thấy."` không bịa gì. Metric tệ nhất chỉ là triệu chứng, cần đọc context mới ra root cause. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` | Combined mode: 1 call gpt-4o-mini/chunk trả JSON (summary, câu hỏi giả định, context, metadata). 103 chunk mất **266.5 s (≈ 2.6 s/chunk)**, bước tốn thời gian nhất pipeline. **Chưa chứng minh được lợi ích:** các câu `context` mẫu mình xem khá chung chung ("Đoạn văn nằm trong tài liệu hướng dẫn về ..."), và mình chưa chạy ablation có/không enrichment nên không biết nó giảm hay tăng retrieval failure. Có fallback heuristic khi không có API key. |

---

## Phần 2: Khó khăn & Cách giải quyết

- **Lỗi kỹ thuật gặp phải (exact error message):**
  - `httpx.ProxyError: 403 Forbidden` (và `curl: (56) CONNECT tunnel failed, response 403`) khi `SentenceTransformer('all-MiniLM-L6-v2')` tải model từ `huggingface.co` — xảy ra trong sandbox cloud ở lần code đầu.
  - `huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled` khi đặt `HF_HUB_OFFLINE=1`.
  - `ModuleNotFoundError: No module named 'qdrant_client'` khi chạy test trước khi cài thư viện.
  - `openai.RateLimitError: Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}` khi chạy lại thí nghiệm lần thứ tư.
  - `python check_lab.py` → `⚠️  pytest error: Command '[... 'pytest', 'tests/', '-v', '--tb=no', '-q']' timed out after 120 seconds`. Xảy ra khi đã có API key nhưng credit đã hết. Chạy `pytest --durations` cho thấy cả 37 test vẫn pass, tổng 252 s, trong đó **riêng `test_m4.py::test_evaluate_returns_metrics` mất 183 s** vì RAGAS retry các call OpenAI bị 429 rồi mới rơi vào `except` (các test còn lại ≤ 9 s). Khi chưa có key, cả bộ chạy xong trong 42 s. Nghĩa là timeout 120 s của `check_lab` là hệ quả của hết credit, không phải lỗi code; mình chưa chạy lại với credit đầy để xác nhận nó xuống dưới 120 s.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Sandbox cloud chỉ cho package manager, chặn HuggingFace/OpenAI (xác nhận bằng `curl` từng host). Lúc đó mình kiểm chứng logic bằng encoder giả (monkeypatch `_SEMANTIC_MODEL`, `_encoder`, `_model`) → 28/37 test pass. Chạy lại trên máy cá nhân có mạng thì 37/37 pass với model thật.
  - Máy có Python 3.14 mặc định nhưng `.python-version` yêu cầu 3.11 (RAGAS cần 3.11+) → tạo venv bằng `python3.11`, vì vậy mọi lệnh đều gọi `.venv/bin/python`.
  - **`evaluate_ragas()` nuốt mọi exception và trả về toàn 0**, nên thiếu API key hay hết credit đều vẫn sinh ra `ragas_report.json` hợp lệ về mặt định dạng nhưng toàn 0, và `check_lab` vẫn báo xanh. Cách xử lý: trước khi chạy cả pipeline, mình gọi thẳng `ragas.evaluate` trên 2 dòng QA tiếng Việt để thấy exception thật (kết quả: 4 metric ra số, không NaN), rồi kiểm tra `num_questions == 20` và điểm khác 0 sau khi chạy. Lỗi hết credit sau đó cho thấy rủi ro là có thật: chạy lại khi hết credit sẽ ghi đè report tốt bằng report toàn 0 (mình đã sao lưu report trước khi thử nghiệm).
  - `recreate_collection` deprecated ở qdrant-client 1.19 → dùng `collection_exists` + `delete_collection` + `create_collection`.
  - Production kém baseline: không đoán mà tách từng yếu tố bằng thí nghiệm cùng retrieval (child/parent × temperature). Trước đó mình in context top-3 của các câu bị `"Không tìm thấy."` để phân biệt lỗi retrieval (laptop) với lỗi generation (PVI).
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Hiểu sai ban đầu rằng điểm thấp luôn là lỗi hệ thống. Đọc mã nguồn RAGAS mới biết Answer Relevancy bằng 0 khi có ý "noncommittal", và faithfulness bị chấm 0.0 cho câu đúng nhưng quá ngắn hoặc có giá trị suy ra (câu nghỉ không lương, câu lương thử việc). Bài học: phải đọc câu trả lời và context của từng câu, không chỉ nhìn điểm.
  - Chưa làm: OCR cho 2 PDF scan (BCTC, Nghị định 13). Không câu nào trong 20 câu test cần chúng, nên bỏ qua; nếu đưa vào dùng thật thì phải OCR.
  - Chưa làm: ablation enrichment, và chưa đưa "return parent" vào pipeline chính thức vì hết credit OpenAI trước khi chạy lại được.

---

## Phần 3: Action Plan cho Project cá nhân

> Mình chưa có thông tin chi tiết về project thật nên giả định một chatbot hỏi đáp chính sách nội bộ (tiếng Việt) giống bài lab; cần chỉnh lại nếu project khác.

### Project: Chatbot hỏi đáp chính sách nội bộ (tiếng Việt)

#### 1. Hiện trạng
- **Pipeline hiện tại:** chunk theo đoạn + dense search + LLM trả lời.
- **Known issues:** nhầm giữa tài liệu cũ và mới (v2023/v2024, mật khẩu v1/v2), bỏ sót từ khóa chính xác, câu hỏi nhiều ý bị thiếu context, chưa có đánh giá định lượng.

#### 2. Kế hoạch áp dụng
1. [ ] **Chunking:** với tài liệu ngắn, chunk theo mục (structure-aware) và **trả về nguyên mục/parent** thay vì child 256 ký tự — bài lab cho thấy child làm mất ngữ cảnh (Faithfulness 0.595 → 0.831 khi trả parent, một lần đo). Chỉ dùng hierarchical khi tài liệu dài vượt vài nghìn ký tự.
2. [ ] **Search:** hybrid BM25 + dense + RRF (BM25 cần tách từ tiếng Việt), **kèm metadata `version`/`effective_date` để lọc tài liệu đã bị thay thế**, vì cả ba tầng (BM25, dense, reranker) đều không tự phân biệt phiên bản.
3. [ ] **Reranking:** bge-reranker-v2-m3, top-20 → top-3; đo lại latency ở trạng thái đã nạp model (hiện 1168 ms/query, chiếm phần lớn thời gian xử lý một câu hỏi); cân nhắc giảm số ứng viên hoặc dùng flashrank nếu cần nhanh.
4. [ ] **Evaluation:** RAGAS trên bộ test 20 câu, nhưng kèm (a) theo dõi tỉ lệ câu trả lời "Không tìm thấy", (b) đặt `temperature=0` khi sinh câu trả lời, (c) sửa harness để **báo lỗi to** thay vì ghi report toàn 0 khi judge hỏng, (d) đọc tay các câu có faithfulness 0.0 để phân biệt lỗi thật với false negative của judge.
5. [ ] **Enrichment:** chỉ giữ combined single-call nếu ablation chứng minh có ích; ưu tiên metadata phiên bản/ngày hiệu lực (rẻ, có tác dụng rõ) hơn là câu `context` do LLM sinh (266 s cho 103 chunk, chưa thấy lợi ích).

#### 3. Timeline
- Tuần 1: dựng baseline + bộ test + RAGAS (có kiểm tra report không toàn 0); đặt `temperature=0`.
- Tuần 2: structure-aware + trả parent + hybrid search, đo lại từng thay đổi một.
- Tuần 3: metadata phiên bản + lọc superseded, rerank, sửa prompt (câu yes/no phủ định, câu cần tính toán); ablation enrichment.
- Tuần 4: query decomposition cho câu nhiều ý, OCR cho PDF scan, phân tích failure lại.
