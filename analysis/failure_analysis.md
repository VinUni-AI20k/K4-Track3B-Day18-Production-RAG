# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phan Duy Bảo (2A202602767)
**Khóa:** K4 - Track 3B

## 0. Phạm vi và nguồn số liệu

Mọi con số trong bảng 1 và bảng 2 đến từ **một lần chạy `python main.py`** (ngày 2026-10-05, gpt-4o-mini sinh câu trả lời, RAGAS 0.1.22 với judge OpenAI mặc định, 20 câu trong `test_set.json`). File gốc: `reports/ragas_report.json`, `reports/naive_baseline_report.json`, `reports/latency_report.json`.

Ba mức bằng chứng được đánh dấu trong từng case:

| Nhãn | Nghĩa |
|---|---|
| **[O]** | Lấy trực tiếp từ lần chạy chính thức (câu hỏi, đáp án, câu trả lời, điểm). |
| **[R]** | Context được **dựng lại** bằng cách chạy lại retrieval trên index enrichment đã cache (`analysis/experiment_parent_vs_child/enrich_cache.json`). Enrichment gọi LLM nên không tất định hoàn toàn, vì vậy context [R] **gần giống chứ không trùng** context của lần chạy chính thức. |
| **[E]** | Thí nghiệm ở mục 4: các biến thể A0/A1/B dùng chung một lượt retrieval trên index đã cache, chỉ đổi một yếu tố. Index này **không trùng** với index của lần chạy chính thức [O]. |

Giới hạn cần biết trước khi đọc:
- `ragas_report.json` chỉ lưu 10 failure thấp nhất, không lưu context từng câu. Vì vậy không thể đối chiếu trực tiếp context chính thức; mục [R] là cách thay thế.
- `evaluate_ragas()` đổi NaN thành 0.0 nên một số ô "0.0" có thể là NaN chứ không phải điểm thật. Report không cho biết số lượng.
- Mỗi cấu hình chỉ chạy 1–2 lần. Độ nhiễu lớn (xem mục 4), nên chênh lệch dưới ~0.1 không nên diễn giải là khác biệt thật.

## 1. Bảng so sánh Naive Baseline vs Production

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8167 | 0.7083 | −0.1083 |
| Answer Relevancy | 0.7261 | 0.6884 | −0.0377 |
| Context Precision | 0.9250 | 0.9458 | +0.0208 |
| Context Recall | 0.9250 | 0.8833 | −0.0417 |

**Production không vượt baseline** trong lần chạy này: chỉ Context Precision tăng nhẹ, ba metric còn lại giảm. Ba metric đạt ≥ 0.70 (Precision, Recall, Faithfulness), nhưng Faithfulness chỉ hơn ngưỡng 0.0083 và không ổn định: chạy lại **cùng code và cấu hình** (nhưng enrichment được sinh lại, nên index không trùng hoàn toàn) cho 0.571 [E, mục 4]. Faithfulness và Answer Relevancy chưa đạt 0.75 (Answer Relevancy 0.688 thấp nhất), nên điều kiện "tất cả metric ≥ 0.75" chưa thỏa.

Baseline đã mạnh một cách đáng ngờ vì corpus rất nhỏ: 26 tài liệu có text (các file markdown dài 11–19 dòng), nên toàn bộ 26 tài liệu đều nằm gọn trong **đúng 1 parent chunk** (26 parents cho 26 tài liệu). Chunk paragraph ≤ 500 ký tự của baseline gần như giữ nguyên cả tài liệu.

## 2. Latency breakdown

Từ `reports/latency_report.json` (cùng lần chạy):

| Bước | Thời gian | Ghi chú |
|---|---|---|
| M1 chunking (26 tài liệu → 103 child) | 0.03 s | |
| M5 enrichment (combined, 1 call/chunk) | 266.5 s | ≈ 2.6 s/chunk, 103 call; chi phí lớn nhất toàn pipeline |
| M2 indexing BM25 + bge-m3 → Qdrant | 17.9 s | Qdrant chạy in-memory (không có container) |
| M3 load reranker | 0.0 s | Lazy-load: model chỉ nạp ở lần `rerank()` đầu tiên |
| M2 hybrid search | 96.9 ms/query | |
| M3 rerank (bge-reranker-v2-m3) | 1167.7 ms/query | **Đã gồm** thời gian nạp model ở query đầu chia đều cho 20 query nên cao hơn latency thuần; chưa đo tách riêng |
| LLM generation (gpt-4o-mini) | 961.4 ms/query | |
| M4 RAGAS (4 metric × 20 câu) | 31.6 s | |

Một query tốn ≈ 2.2 s, trong đó rerank là phần lớn. Latency của baseline không được đo nên không có so sánh.

## 3. Bottom-5 Failures (theo điểm trung bình thấp nhất trong `ragas_report.json`)

### #1 — "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"
- **[O] Điểm:** trung bình 0.4167, metric tệ nhất = faithfulness 0.0.
- **[O] Expected:** 30 triệu nằm trong khoảng 5–50 triệu → Giám đốc phòng ban phê duyệt; cần xác nhận cấu hình kỹ thuật của phòng CNTT.
- **[O] Got:** `"Không tìm thấy."`
- **[R] Context top-3 sau rerank:** chunk "cần xác nhận của phòng CNTT" (`mua_sam.md`, 0.781) + 2 chunk nhiễu (`tam_ung.md` 0.053, `hoan_chi_dao_tao.md` 0.043). **Chunk bảng phê duyệt theo hạn mức của `mua_sam.md` không có trong top-3.**
- **[E]** Khi trả cả tài liệu (parent): câu trả lời đúng, faithfulness 1.0, recall 1.0. Với child + temperature 0, mô hình nói "Giám đốc phòng ban" dù context không có thông tin đó (faithfulness 0.33).
- **Error Tree:** Output đúng? **Không** → Context đúng? **Không** (thiếu nửa "ai phê duyệt") → Query OK? Có, nhưng là câu 2 vế (multi-hop trong cùng một tài liệu) → **Root cause: chunk con 256 ký tự tách bảng phê duyệt khỏi câu xác nhận CNTT; top-3 bị 2 chunk nhiễu chiếm chỗ.**
- **Diagnosis:** Missing relevant chunks (context_recall) dẫn tới refusal.
- **Fix:** trả parent thay vì child (đã thử, xem mục 4); hoặc tăng số chunk sau rerank cho câu multi-hop.

### #2 — "Bao lâu phải đổi mật khẩu một lần?"
- **[O] Điểm:** trung bình 0.4583, metric tệ nhất = faithfulness 0.0.
- **[O] Expected:** 120 ngày theo chính sách v2.0 hiện hành (v1.0 là 90 ngày, đã bị thay thế).
- **[O] Got:** `"Không tìm thấy."`
- **[R] Context top-3:** `mat_khau_v1.md` ("mỗi 90 ngày", rerank 0.966), `mat_khau_v2.md` ("mỗi 120 ngày", rerank 0.966), rồi đoạn "đã được thay thế bởi v2.0" của v1 (0.105). Hai phiên bản **đồng điểm** nên reranker không phân biệt được bản hiện hành.
- **[E]** Child + temperature 0 vẫn từ chối; trả parent thì trả lời đúng "mỗi 120 ngày", faithfulness 1.0 (nhưng context precision chỉ 0.5 vì kéo cả tài liệu v1 vào).
- **Error Tree:** Output đúng? **Không** → Context đúng? **Có một phần** (có fact đúng nhưng lẫn bản cũ mâu thuẫn) → Query OK? Có, ngắn và rõ → **Root cause: index không có metadata phiên bản/ngày hiệu lực, và prompt "chỉ dựa trên context" không có quy tắc giải quyết mâu thuẫn giữa hai phiên bản, nên LLM từ chối.**
- **Diagnosis:** nhầm phiên bản tài liệu (version confusion) + prompt không xử lý mâu thuẫn.
- **Fix:** gắn `version`/`effective_date`/`is_current` khi enrich và lọc hoặc ưu tiên bản hiện hành ở bước search; thêm vào prompt "nếu context có nhiều phiên bản, dùng bản có ngày hiệu lực mới nhất".

### #3 — "Nghỉ phép không lương 20 ngày cần ai phê duyệt?"
- **[O] Điểm:** trung bình 0.5302, metric tệ nhất = faithfulness 0.0.
- **[O] Expected:** nghỉ 16–30 ngày cần Giám đốc điều hành (CEO); nghỉ trên 14 ngày nhân viên tự đóng bảo hiểm.
- **[O] Got:** `"Giám đốc điều hành (CEO)."` — **đúng nội dung**.
- **[R] Context top-1** (`nghi_phep_khong_luong.md`, 0.918) chứa nguyên văn "Nghỉ từ 16-30 ngày: cần phê duyệt của Giám đốc điều hành (CEO)".
- **Error Tree:** Output đúng? **Đúng** → Context đúng? **Đúng** → Query OK? Có → **Root cause nhiều khả năng nằm ở metric, không phải RAG:** câu trả lời một cụm từ không nêu lại chủ ngữ ("nghỉ 20 ngày"), nên bước tách phát biểu của RAGAS không có gì để đối chiếu với context. *(Đây là giả thuyết, chưa kiểm chứng trực tiếp; điểm gián tiếp: khi mô hình trả lời thành câu đầy đủ ở [E] thì faithfulness lên 0.5.)*
- **Diagnosis:** LLM hallucinating (theo bảng chẩn đoán) nhưng thực chất là false negative của faithfulness.
- **Fix:** prompt yêu cầu trả lời bằng câu đầy đủ, nêu lại điều kiện ("Nghỉ không lương 20 ngày thuộc mức 16–30 ngày nên cần CEO phê duyệt").

### #4 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
- **[O] Điểm:** trung bình 0.625, metric tệ nhất = answer_relevancy 0.0.
- **[O] Expected:** 15 + 3 = 18 ngày (v2024); lương Senior 20–35 triệu.
- **[O] Got:** "18 ngày phép (15 + 3). Không tìm thấy thông tin cụ thể về lương trong context." — nửa đầu đúng, nửa sau thiếu.
- **Quy tắc RAGAS (đã kiểm chứng trong mã nguồn `ragas/metrics/_answer_relevance.py`):** `score = cosine_sim.mean() * int(not committal)`; chỉ cần câu trả lời bị gắn "noncommittal" là điểm về **0**, dù một phần câu trả lời đúng.
- **[E]** Cả child lẫn parent đều có recall 0.5 cho câu này: top-3 không phủ được cả `nghi_phep_nam_v2024.md` lẫn `bang_luong_2024.md`.
- **Error Tree:** Output đúng? **Một nửa** → Context đúng? **Không** (thiếu bảng lương) → Query OK? **Không tối ưu:** câu ghép hai ý khác tài liệu → **Root cause: một truy vấn + top-3 không đủ phủ hai tài liệu.**
- **Diagnosis:** Answer doesn't match question do thiếu context cho vế 2; metric phạt nặng câu trả lời thiếu.
- **Fix:** query decomposition (tách thành 2 truy vấn con, merge context) hoặc tăng top-k.

### #5 — "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **[O] Điểm:** trung bình 0.6651, metric tệ nhất = faithfulness 0.0.
- **[O] Expected:** hạn 15 ngày, quá hạn 5 ngày, phí 2%/tháng × 15.000.000 = 300.000/tháng (≈ 50.000 cho 5 ngày).
- **[O] Got:** "Phạt = 2% × 15.000.000 = 300.000 VNĐ" — bỏ qua hạn 15 ngày và đơn vị "/tháng", không tính theo số ngày quá hạn.
- **[E]** Cả child lẫn parent đều nhắc đúng "hạn 15 ngày" và "2%/tháng" nhưng faithfulness vẫn chỉ 0.40–0.45: bước suy luận (tính phí theo ngày) không có trong context.
- **Error Tree:** Output đúng? **Không** → Context đúng? **Phần lớn đúng** (có 2% và hạn 15 ngày) → Query OK? Có → **Root cause: bước generation, không phải retrieval:** phải tự suy luận số ngày quá hạn và tính phí tỉ lệ; gpt-4o-mini làm tắt. (Đáp án mẫu cũng mơ hồ giữa 300.000/tháng và 50.000/5 ngày, nên điểm bị nhiễu.)
- **Diagnosis:** LLM hallucinating / suy luận không được context hỗ trợ.
- **Fix:** prompt yêu cầu liệt kê dữ kiện trích từ context rồi mới tính; `temperature=0`.

### Các failure còn lại trong top-10 (tóm tắt)

| # | Câu hỏi | Metric tệ nhất | Nhận xét |
|---|---|---|---|
| 6 | Lương thử việc Junior mức cao nhất | faithfulness 0.0 (avg 0.706) | Trả 17.000.000 (khớp con số cuối của đáp án). Giá trị suy ra (85% × 20 triệu) không có nguyên văn trong context nên judge chấm 0 — cùng loại với #3/#5. |
| 7 | Thử việc có được hưởng bảo hiểm PVI không? | answer_relevancy 0.0 (avg 0.75) | `"Không tìm thấy."` dù context top-1 ghi rõ "chưa được hưởng gói bảo hiểm sức khỏe PVI" [R]. Còn từ chối ngay cả khi trả cả tài liệu [E] → **lỗi prompt/generation**: câu hỏi yes/no mà đáp án là phủ định, mô hình không suy ra "Không". |
| 8 | Thông tin lương thuộc cấp độ dữ liệu nào? | context_recall 0.5 (avg 0.842) | Đáp án đúng ("Bí mật") nhưng ground_truth có thêm quy tắc mã hóa/hạn chế của cấp 3, nằm ở tài liệu khác. |
| 9 | Có cần kích hoạt MFA không? | context_recall 0.5 (avg 0.849) | Câu trả lời đúng; phần thiếu là so sánh với v1.0 (không yêu cầu MFA) — không ảnh hưởng đáp án. |
| 10 | Tài trợ khóa học 25 triệu, nghỉ sau 8 tháng | faithfulness 0.5 (avg 0.851) | Trả 100% = 25 triệu (đúng) nhưng không nêu điều kiện cam kết 1 năm. |

**Quy luật rút ra từ top-10 [O]:** 3/10 câu là `"Không tìm thấy."` (laptop, mật khẩu, PVI); 6/10 có metric tệ nhất là faithfulness (5 trong số đó bằng 0.0). Trong nhóm này, #3 và #6 là câu trả lời **đúng** mà judge chấm 0.0 vì câu quá ngắn hoặc có giá trị suy ra; #5 thì sai thật (lỗi suy luận); #1, #2 là lỗi retrieval/phiên bản thật. Vậy điểm Faithfulness 0.708 vừa **đánh giá thấp** chất lượng thật (false negative của judge), vừa bị kéo xuống bởi lỗi thật.

## 4. Thí nghiệm bổ sung: child vs parent, temperature (CHƯA áp dụng vào pipeline)

Mục đích: tách xem lỗi đến từ retrieval (child quá nhỏ) hay từ generation. Ba biến thể A0, A1, B dùng **chung một lượt retrieval** (hybrid → rerank top-3 trên một index đã enrich và cache), chỉ đổi cách tạo context và `temperature`, nên so sánh A0/A1/B với nhau là so sánh cùng retrieval. **Dòng "lần chạy chính thức" thì khác:** nó dùng index enrichment của lần `main.py` riêng, không phải index đã cache này, nên không so sánh trực tiếp từng câu được. Script và kết quả thô: `analysis/experiment_parent_vs_child/`.

| Biến thể | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | "Không tìm thấy" |
|---|---|---|---|---|---|
| Lần chạy chính thức (mục 1, child, temperature mặc định) | 0.708 | 0.688 | 0.946 | 0.883 | ≥ 3/20 (chỉ biết 3 trong top-10) |
| A0 — cấu hình hiện tại, chạy lại | 0.571 | 0.593 | 0.942 | 0.867 | 6/20 |
| A1 — child + `temperature=0` | 0.595 | 0.635 | 0.942 | 0.867 | 5/20 |
| **B — parent + `temperature=0`** | **0.831** | **0.764** | 0.950 | 0.925 | **1/20** |

Đọc kết quả (có giới hạn):
- A0 so với lần chính thức: cùng code và cấu hình, Faithfulness lệch 0.137. Đây là mức nhiễu tổng hợp của thiết lập hiện tại, từ ba nguồn không tách được: temperature mặc định khi sinh câu trả lời, judge LLM của RAGAS, và **enrichment không tất định** (sinh lại enrichment cho ra context khác nên retrieval cũng đổi). Mọi chênh lệch nhỏ hơn mức này không đáng tin. Riêng so sánh A1 với B là cùng retrieval nên không chịu nguồn nhiễu enrichment.
- `temperature=0` một mình (A0 → A1) gần như không giúp.
- Trả **parent** (A1 → B) là yếu tố tạo khác biệt: Faithfulness +0.24, Answer Relevancy +0.13, Recall +0.06, và số câu từ chối giảm từ 5 xuống 1. Chênh lệch này lớn hơn mức nhiễu đã đo, và khớp với cơ chế đã thấy ở case #1, #2 (child làm mất ngữ cảnh). Đây cũng là đúng thiết kế "retrieve child → return parent" của M1, mà pipeline hiện chỉ index child và trả nguyên child.
- **Chưa lặp lại:** lượt chạy lại biến thể B bị dừng vì tài khoản OpenAI hết credit (`credit_balance_exhausted`). B chỉ có **một** lần đo, và chưa được đưa vào `src/pipeline.py`, nên các số ở mục 1 vẫn là của pipeline hiện tại. Nếu B giữ được mức này thì cả 4 metric đạt ≥ 0.75.
- Câu PVI (#7) vẫn từ chối ở B: parent không sửa được lỗi prompt.

## 5. Vì sao production không hơn baseline (giả thuyết xếp theo mức tin cậy)

1. **Chunk child 256 ký tự trên corpus vốn đã nhỏ** làm mất ngữ cảnh mà baseline giữ được (bằng chứng [E], mục 4; probe BM25 offline cũng cho child kém basic: 0.664 vs 0.742).
2. **Không phân biệt phiên bản tài liệu** (v2023/v2024, mật khẩu v1/v2): cả BM25, dense lẫn reranker đều coi hai phiên bản là ứng viên ngang nhau (#2).
3. **Generation không ổn định:** không đặt `temperature`, prompt "chỉ dựa trên context" khiến mô hình từ chối thay vì suy luận (#2, #7).
4. **Enrichment chưa chứng minh được giá trị:** trong các mẫu đã xem, câu `context` do LLM sinh khá chung chung ("Đoạn văn nằm trong tài liệu hướng dẫn về ..."), ít thông tin hơn tiêu đề tài liệu. Tôi **chưa làm ablation** (chạy có/không enrichment) nên không thể khẳng định enrichment tốt hay xấu; chi phí 266 s cho 103 chunk là có thật.
5. Corpus nhỏ + 20 câu nên độ nhiễu cao; các kỹ thuật production (hybrid, rerank) có lợi hơn khi corpus lớn.

## 6. Probe offline BM25 (bằng chứng phụ, đo từ trước khi có API)

`analysis/offline_bm25_probe.py` — BM25-only, top-3, không rerank/enrichment, đo tỉ lệ dữ kiện số của ground_truth xuất hiện trong context (proxy cho context recall, **không phải RAGAS**). Số chunk đã được đối chiếu lại trong lần chạy này (khớp).

| Chunking | Số chunk | Fact-coverage@3 |
|---|---|---|
| basic (500 ký tự) | 57 | 0.742 |
| hierarchical child (256) | 103 | 0.664 |
| structure-aware | 107 | 0.685 |
| semantic (ngưỡng 0.85, all-MiniLM-L6-v2) | 208 | chưa đo |

Probe dựa trên số nên có nhiễu (ví dụ Q8 MFA bị chấm 0.00 dù retrieval đúng).

## 7. Case Study (presentation): "Bao lâu phải đổi mật khẩu một lần?"

**Error Tree walkthrough:**
1. **Output đúng?** Không — pipeline trả `"Không tìm thấy."` trong khi đáp án là 120 ngày.
2. **Context đúng?** Có một phần — top-3 chứa cả "mỗi 120 ngày" (v2) lẫn "mỗi 90 ngày" (v1); reranker chấm hai chunk **bằng nhau** (0.966).
3. **Query rewrite OK?** Có — câu hỏi ngắn và rõ; vấn đề không nằm ở câu hỏi.
4. **Fix ở bước nào?** Hai chỗ: (a) **indexing:** gắn metadata `version`/`effective_date` và lọc hoặc boost bản hiện hành trước khi rerank; (b) **generation:** thêm quy tắc "ưu tiên phiên bản mới nhất" vào prompt thay vì để mô hình từ chối. Thí nghiệm [E] cho thấy chỉ riêng việc trả cả tài liệu cũng đủ để mô hình chọn đúng 120 ngày, nhưng đổi lại Context Precision của câu này giảm còn 0.5.

**Nếu có thêm 1 giờ (theo thứ tự ưu tiên):**
1. Đưa "retrieve child → return parent" vào `src/pipeline.py`, đặt `temperature=0`, chạy lại `main.py` và xác nhận B có lặp lại được không (cần credit OpenAI).
2. Thêm metadata phiên bản và lọc tài liệu superseded ở bước search.
3. Sửa prompt: trả lời bằng câu đầy đủ, xử lý câu hỏi yes/no phủ định (#7), liệt kê dữ kiện trước khi tính (#5).
4. Query decomposition cho câu ghép nhiều ý (#4).
5. Ablation enrichment để biết 266 s đó có đáng không.
6. Chặn `evaluate_ragas()` ghi report toàn 0 khi judge lỗi (thấy rủi ro này khi hết credit: lỗi 429 bị nuốt và report sẽ bị ghi đè bằng 0).
