# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phan Duy Bảo (2A202602767)
**Khóa:** K4 - Track 3B

> **Phạm vi dữ liệu (đọc trước):** môi trường cloud chặn `huggingface.co` và `api.openai.com`, nên **chưa có điểm RAGAS thật**
> (bge-m3, reranker, LLM judge). Bảng RAGAS bên dưới để `⏳`, điền sau khi chạy `python main.py` ở máy có mạng + `OPENAI_API_KEY`.
> Phần phân tích dưới đây dựa trên **probe retrieval offline BM25-only** (`analysis/offline_bm25_probe.py`, top-3, không rerank,
> không enrichment): đo tỉ lệ dữ kiện số của ground_truth xuất hiện trong context. Đây là proxy cho *context recall*, **không phải RAGAS**,
> và chỉ phản ánh tầng BM25 — pipeline đầy đủ (dense + RRF + rerank) có thể tốt hơn.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | ⏳ | ⏳ | ⏳ |
| Answer Relevancy | ⏳ | ⏳ | ⏳ |
| Context Precision | ⏳ | ⏳ | ⏳ |
| Context Recall | ⏳ | ⏳ | ⏳ |

### Probe offline (BM25, top-3, fact-coverage trung bình trên 20 câu)

| Chunking | Số chunk | Fact-coverage@3 |
|---|---|---|
| basic (500 ký tự) | 57 | 0.742 |
| hierarchical child (256) | 103 | 0.664 |
| structure-aware | 107 | 0.685 |

Nhận xét: chunk nhỏ hơn **không** tự động tốt hơn với BM25 — child 256 ký tự tách rời dữ kiện khỏi ngữ cảnh (đặc biệt câu cần 2 dữ kiện cạnh nhau),
nên coverage thấp hơn basic. Đây là lý do pipeline nên trả **parent** thay vì child, hoặc dùng enrichment để gắn ngữ cảnh. Cần kiểm chứng lại bằng RAGAS.

## Bottom-5 Failures (theo probe, chunking basic)

### #1 — Q4: "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
- **Expected:** 15 ngày theo v2024 (v2023 = 12 ngày đã bị thay thế).
- **Got (top-3 BM25, chunking basic):** `nghi_phep_dac_biet.md`, `nghi_phep_nam_v2023.md` (bản cũ), `nghi_phep_khong_luong.md`; **không có bản v2024 hiện hành**.
- **Worst metric:** context_recall (coverage 0.00).
- **Error Tree:** Output sai → Context đúng? **Không** (bản cũ, thiếu bản mới) → Query OK? Có (query ngắn, rõ) → Root cause: BM25 không phân biệt phiên bản; từ khóa "nghỉ phép" khớp mọi tài liệu nghỉ.
- **Suggested fix:** thêm metadata `version/effective_date` và ưu tiên/loại bản superseded; dense + rerank để đẩy v2024 lên; enrichment ghi rõ "phiên bản hiện hành".

### #2 — Q18: "Lương thử việc Junior mức cao nhất là bao nhiêu?"
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ.
- **Got:** `thu_viec.md`, `bang_luong_2024.md` (đoạn "Lương thử việc 85%") nhưng **không có chunk bảng lương** chứa "Junior 12–20 triệu".
- **Worst metric:** context_recall (0.33).
- **Error Tree:** Output sai → Context đúng? Thiếu một nửa (multi-hop: tỉ lệ 85% ở 1 chunk, mức lương ở chunk khác) → Query OK? Có → Root cause: dữ kiện cần ghép nằm ở 2 chunk khác nhau, top-3 không phủ cả hai.
- **Suggested fix:** tăng top-k sau rerank cho câu multi-hop; trả parent (đoạn lớn hơn); HyQA enrichment để chunk bảng lương khớp câu hỏi về "lương Junior".

### #3 — Q17: "Tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** quá hạn 5 ngày, 2%/tháng × 15.000.000 = 300.000/tháng (~50.000 cho 5 ngày).
- **Got:** `tam_ung.md` (2 chunk, có 15 ngày + 2%/tháng) và 1 chunk `cong_tac_phi.md` không liên quan.
- **Worst metric:** context_precision (chunk thứ 3 nhiễu) + answer cần phép tính.
- **Error Tree:** Output sai → Context đúng? Phần lớn đúng → Query OK? Có → Root cause: retrieval ổn; rủi ro nằm ở **generation** (phải tự tính pro-rata) → faithfulness/relevancy.
- **Suggested fix:** rerank loại chunk `cong_tac_phi`; prompt yêu cầu trình bày phép tính từ dữ kiện trong context.

### #4 — Q12: "Senior 9 năm thâm niên được nghỉ bao nhiêu ngày và lương trong khoảng nào?"
- **Expected:** 15 + 3 = 18 ngày (v2024); lương Senior 20–35 triệu.
- **Got:** `nghi_phep_khong_luong.md`, `nghi_phep_nam_v2024.md`, `nghi_phep_nam_v2023.md` — **thiếu `bang_luong_2024.md`**, lại lẫn bản v2023.
- **Worst metric:** context_recall (0.50) và nhầm phiên bản.
- **Error Tree:** Output sai → Context đúng? Không (thiếu bảng lương, lẫn bản cũ) → Query OK? Câu gộp 2 ý → Root cause: câu hỏi 2 vế, một top-k không đủ phủ.
- **Suggested fix:** query decomposition (tách 2 truy vấn) hoặc tăng top-k; lọc phiên bản.

### #5 — Q14: "Tài trợ khóa học 25 triệu, nghỉ sau 8 tháng. Phải hoàn trả bao nhiêu?"
- **Expected:** cam kết 1 năm; hoàn trả theo tỉ lệ thời gian còn thiếu.
- **Got:** `hoan_chi_dao_tao.md` (2 chunk) + `dao_tao_noi_bo.md`.
- **Worst metric:** context_recall (0.50) — chunk chứa bảng/công thức hoàn trả có thể bị cắt khỏi chunk nêu "1 năm".
- **Error Tree:** Output sai → Context đúng? Chủ yếu đúng tài liệu → Query OK? Có → Root cause: chunking tách điều kiện và công thức; cần tính toán.
- **Suggested fix:** structure-aware/parent chunk giữ nguyên mục; enrichment context-prepend.

**Lưu ý trung thực về probe:** Q8 (MFA) bị chấm 0.00 nhưng thực tế retrieval đúng — `mat_khau_v2.md` mục MFA đứng đầu; 2 số "thiếu" (10, 20) là số trong ground_truth không liên quan tới dữ kiện chính. Probe chỉ dựa trên số nên có nhiễu; các case #1, #2, #4 là lỗi retrieval thật khi đọc trực tiếp chunk.

## Case Study (presentation)

**Question:** Q4 — "Nhân viên được nghỉ bao nhiêu ngày phép năm?"

**Error Tree walkthrough:**
1. Output đúng? → Sai/nguy cơ sai: nếu trả theo context top-3 sẽ ra 12 ngày (v2023, đã bị thay thế).
2. Context đúng? → Không: bản v2024 hiện hành không nằm trong top-3 BM25.
3. Query rewrite OK? → Có; vấn đề không nằm ở câu hỏi mà ở việc không phân biệt phiên bản.
4. Fix ở bước: **retrieval/indexing** — gắn metadata phiên bản + ngày hiệu lực khi chunk/enrich, lọc hoặc boost bản hiện hành, rồi để reranker chốt.

**Nếu có thêm 1 giờ:** thêm metadata `is_current` từ trường "Phiên bản/Ngày hiệu lực" trong header tài liệu, lọc ở bước search; thử trả parent chunk; chạy RAGAS để xác nhận hiệu quả (⏳).
