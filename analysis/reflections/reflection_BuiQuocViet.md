# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Bùi Quốc Việt  
**Mã số sinh viên:** 2A202602884  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026 (GMT+7)

Bài thực hành được triển khai và kiểm tra với sự hỗ trợ của Codex. Nội dung dưới đây ghi lại các thay đổi và lỗi thực tế trong quá trình làm bài; phần kế hoạch là đề xuất ứng dụng tiếp theo.

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Khái niệm | Module | Hàm cụ thể | Quan sát và phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu, dùng MiniLM để so sánh cosine giữa hai câu liền nhau, chuyển sang chunk mới khi thấp hơn ngưỡng 0,85. Đây là một chiến lược lựa chọn; pipeline chính dùng phân cấp. |
| Parent-child chunking | M1 | `chunk_hierarchical()` | Đoạn con tối đa 256 ký tự phục vụ tìm kiếm; đoạn cha tối đa 2.048 ký tự cung cấp ngữ cảnh. `parent_id` có nguồn tài liệu để tránh trùng giữa các file. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Chia theo tiêu đề Markdown và không nhận nhầm dòng bắt đầu bằng `#` trong khối code là tiêu đề. |
| Hybrid Search | M2 | `BM25Search`, `DenseSearch` | BM25 xử lý từ khóa tiếng Việt; bge-m3 tạo vector lưu trên Qdrant. Hai bộ tìm kiếm bổ sung cho nhau. |
| Rank fusion | M2 | `reciprocal_rank_fusion()` | Gộp thứ hạng bằng công thức 1/(60 + rank + 1), không cộng trực tiếp điểm BM25 và cosine vì thang đo khác nhau. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Chấm cặp câu hỏi–đoạn văn và lấy top-3. Kiểm thử xác nhận đoạn nghỉ phép đứng trước đoạn bảo mật và thử việc. Độ trễ phụ thuộc CPU/GPU; chưa kết luận đã đạt mục tiêu 150 ms. |
| Evaluation | M4 | `evaluate_ragas()` | Chấm faithfulness, answer relevancy, context precision, context recall. Groq chỉ hỗ trợ n=1 nên answer relevancy dùng strictness=1. Mỗi prompt chấm giữ một ví dụ mẫu để giảm token; cả baseline và Production dùng cùng cấu hình. Kết quả cần được đọc cùng cấu hình này. |
| Failure analysis | M4 | `failure_analysis()` | Xếp câu hỏi theo trung bình bốn điểm, tìm chỉ số thấp nhất để gợi ý khâu cần kiểm tra; gợi ý tự động cần đối chiếu thủ công với câu trả lời và tài liệu. |
| Enrichment | M5 | `_enrich_single_call()`, `enrich_chunks()` | Một lần gọi AI sinh summary, questions, context và metadata. Chúng được đưa vào văn bản lập chỉ mục; nội dung gốc và metadata nguồn được giữ lại. |
| Grounded generation | Pipeline | `run_query()` | Sau khi tìm và rerank đoạn con, lấy đoạn cha gốc cho mô hình trả lời, tránh dùng câu hỏi sinh thêm như bằng chứng. |

Kho có 28 file nhưng bộ đọc chỉ dùng 26 tài liệu có văn bản: hai PDF scan không có text layer được bỏ qua và cần OCR trước khi lập chỉ mục. Baseline tạo 57 đoạn; pipeline phân cấp tạo 121 đoạn con. Không thể coi số lượng đoạn tăng là bằng chứng tự động rằng chất lượng trả lời tăng.

## Phần 2: Khó khăn và cách giải quyết (Challenges & Debugging)

### 1. Đánh giá đứng mãi ở 0/80

- **Hiện tượng:** mô hình đã nạp đủ weights nhưng thanh `Evaluating` không tiến triển.
- **Nguyên nhân:** RAGAS 0.1.x tạo `asyncio.as_completed` trước `asyncio.run`; trên Python 3.13, tác vụ được gắn ngay với vòng lặp cũ, trong khi chương trình chờ trên vòng lặp khác.
- **Cách xử lý:** áp dụng `nest_asyncio` trên Python 3.13 để tái sử dụng vòng lặp. Thêm regression test cho hai tác vụ Executor nhỏ để kiểm tra chúng thật sự hoàn tất.

### 2. Chuyển nhà cung cấp API

- **Lỗi:** `InternalServerError; HTTP status=503` khi chấm bằng Gemini.
- **Cách xử lý:** chuyển sang endpoint tương thích OpenAI của Groq. Việc chuyển phải gồm tên biến khóa, base URL, model, baseline, enrichment và bộ chấm, không chỉ đổi khóa trong `.env`.
- **Lỗi model:** `NotFoundError; HTTP status=404` với model Llama không có trong tài khoản. Đọc danh sách model thật và chọn `openai/gpt-oss-120b` do Groq cung cấp.
- **Điểm tương thích:** RAGAS mặc định có thể yêu cầu nhiều câu hoàn thành. Groq chỉ chấp nhận n=1 nên cấu hình `AnswerRelevancy(strictness=1)`.

### 3. Hạn mức Groq

- **Lỗi:** `RateLimitError; HTTP status=429` khi chạy bộ đánh giá dài.
- **Cách xử lý:** đọc các header `x-ratelimit-remaining-tokens` và `x-ratelimit-reset-tokens`, chờ khi ngân sách token chưa đủ; giới hạn tác vụ chấm đồng thời và bật retry có giới hạn. Lưu cache enrichment và kết quả chấm từng câu để có thể tiếp tục sau gián đoạn. Không ghi điểm 0 của một lần gọi thất bại thành kết quả chấm thật.

### 4. Kiểm thử và dữ liệu

- **Cảnh báo:** `PytestCacheWarning` do không có quyền ghi cache trong môi trường thực thi. Chạy với `-p no:cacheprovider` để tránh cảnh báo này.
- **PDF scan:** bộ đọc pypdf không trích được chữ từ ảnh. Giữ cảnh báo, không giả định hai file đó đã được xử lý. OCR là bước cải tiến tiếp theo.
- **Phân biệt test với đánh giá:** unit test kiểm tra logic và định dạng; demo một câu chỉ chứng minh khả năng ghép module. Chất lượng toàn bộ cần báo cáo thật trên đủ 20 câu.

### Kiến thức cần bổ sung

Cần hiểu rõ hơn về hiệu lực văn bản, giới hạn token, bất đồng bộ Python và sự khác biệt giữa đánh giá dựa trên tài liệu với đánh giá đúng theo chính sách hiện hành. Một câu có faithfulness cao vẫn có thể sai nếu dựa vào tài liệu cũ.

## Phần 3: Action Plan cho project cá nhân

### Project đề xuất: Trợ lý tra cứu quy chế tiếng Việt

#### 1. Hiện trạng

Dùng pipeline của lab làm nguyên mẫu: đọc Markdown/PDF có text, hierarchical chunking, enrichment, Hybrid Search, reranking và trả lời có ngữ cảnh. Các rủi ro cần kiểm tra là văn bản cũ lẫn mới, thiếu OCR và giới hạn API. Đây là đề xuất tiếp tục từ lab, không phải một hệ thống đã triển khai cho người dùng thật.

#### 2. Kế hoạch cải tiến

1. **Chunking:** dùng cấu trúc tiêu đề kết hợp cha–con để tránh tách điều kiện áp dụng; thử nhiều kích thước trên cùng bộ câu hỏi.
2. **Search:** giữ BM25 + Dense + RRF; lưu loại văn bản, phiên bản, ngày hiệu lực và trạng thái bị thay thế để lọc đúng quy định.
3. **Reranking:** giữ bge-reranker-v2-m3; đo độ trễ sau warm-up và thử Flashrank nếu tài nguyên hạn chế. Không suy ra ưu thế tốc độ khi chưa đo.
4. **Evaluation:** duy trì bộ 20 câu hiện tại, bổ sung câu hỏi không có đáp án và câu hỏi xung đột phiên bản. Dùng cả bốn chỉ số RAGAS và kiểm tra thủ công nội dung trả lời.
5. **Enrichment:** dùng combined mode, lưu cache theo hash nội dung để tránh gọi lại; kiểm tra summary và metadata không làm thay đổi số liệu gốc.
6. **Vận hành:** thêm OCR cho PDF scan, quản lý khóa bằng biến môi trường, ghi thời gian và số lần gọi API, lưu kết quả trung gian để tiếp tục khi bị giới hạn dịch vụ.

#### 3. Timeline triển khai

- **Tuần 1:** bổ sung OCR và metadata hiệu lực; mở rộng bộ câu hỏi; xác định điểm baseline.
- **Tuần 2:** thử các chiến lược chunking, đo retrieval và reranking; phân tích lỗi, tối ưu cache/API rồi làm giao diện thử nghiệm có trích nguồn.

Tiêu chí nghiệm thu là có báo cáo đánh giá hợp lệ, giải thích được các ca sai và chứng minh cải tiến trên cùng dữ liệu/cấu hình. Mục tiêu tăng điểm không thay thế kiểm tra tính đúng của câu trả lời.

Trong lần chạy hoàn chỉnh, M5 dùng GPT-OSS 20B để chuẩn bị cache song song với baseline; tạo câu trả lời và bộ chấm vẫn dùng GPT-OSS 120B cho cả hai pipeline. Model enrichment được lưu trong metadata để truy vết cấu hình.
