# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phan Duy Bảo (2A202602767)
**Khóa:** K4 - Track 3B

> **Trạng thái: CHƯA CÓ SỐ LIỆU THẬT.** Môi trường cloud chặn `huggingface.co` và `api.openai.com` nên chưa chạy được
> embedding bge-m3, reranker và RAGAS. Bảng và bottom-5 dưới đây được điền sau khi chạy `python main.py` ở máy có mạng
> và `OPENAI_API_KEY`; `reports/ragas_report.json` sẽ chứa danh sách bottom-5 do `failure_analysis()` sinh ra.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | ⏳ | ⏳ | ⏳ |
| Answer Relevancy | ⏳ | ⏳ | ⏳ |
| Context Precision | ⏳ | ⏳ | ⏳ |
| Context Recall | ⏳ | ⏳ | ⏳ |

## Bottom-5 Failures

Mỗi mục điền theo mẫu (lấy từ `failures` trong `reports/ragas_report.json`):

### #1 … #5
- **Question:** ⏳
- **Expected:** ⏳
- **Got:** ⏳
- **Worst metric:** ⏳
- **Error Tree:** Output sai → Context đúng? → Query OK? → Root cause: ⏳
- **Suggested fix:** ⏳

## Giả thuyết cần kiểm chứng (rút ra từ việc đọc corpus và test_set, chưa đo)

1. **Nhầm phiên bản:** `nghi_phep_nam_v2023.md` (12 ngày) và `v2024.md` (15 ngày), `mat_khau_v1.md` (90 ngày) và `v2.md` (120 ngày) cùng xuất hiện ở top-3 khi query "nghỉ phép năm" (đã thấy với encoder giả). Nếu câu "version" trong test_set tụt điểm, lỗi nằm ở retrieval (context đúng nhưng lẫn bản cũ) → fix: thêm metadata `version/effective_date` và lọc, hoặc ưu tiên bản mới.
2. **Câu phủ định / multi-hop:** cần context từ nhiều chunk; child 256 ký tự có thể quá nhỏ → context_recall thấp → fix: trả parent thay vì child.
3. **PDF scan:** BCTC.pdf và Nghị định 13 không được index (không có text layer). Câu hỏi về hai tài liệu này sẽ fail ở context_recall → fix: OCR trước khi chunk.

## Case Study (presentation)

⏳ Chọn 1 câu từ bottom-5 sau khi có số liệu, đi theo Error Tree:
1. Output đúng? →
2. Context đúng? →
3. Query rewrite OK? →
4. Fix ở bước:

**Nếu có thêm 1 giờ:** ⏳
