# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Bùi Quốc Việt  
**Mã số sinh viên:** 2A202602884  
**Khóa:** K4 - Track 3B  

---

## Phạm vi và bằng chứng

Baseline đã chấm thành công trên 20 câu. Production có câu trả lời và context cho 20 câu nhưng RAGAS chưa hoàn tất do `RateLimitError`; báo cáo ghi `evaluation_status: failed`, `num_questions: 0`. Bốn điểm 0 là placeholder, không phải điểm chất lượng.

Báo cáo này phân tích **bottom-5 Baseline**, xếp tăng dần theo trung bình bốn metric, và đối chiếu thủ công với Production trên cùng câu hỏi. **Chưa xác định được bottom-5 Production** theo yêu cầu đề bài; cần cập nhật khi đánh giá hoàn tất. Một ca điểm thấp có thể là lỗi câu trả lời hoặc bất đồng với bộ chấm, không tự động là hallucination.

Bằng chứng: `reports/naive_baseline_report.json`, `test_set.json`, tài liệu gốc trong `data/` và dữ liệu Production lưu tại `.run_cache/evaluation/inputs_cf20169f9c4c7f1c326f241565c097edfcab00908b7347d5339e7463904b4253.json`. Cache không đưa lên Git; các kết quả cần thiết được trích lại dưới đây.

Bộ chấm dùng `openai/gpt-oss-120b`, answer relevancy strictness=1, một ví dụ mỗi prompt, embedding bge-m3 cục bộ. Không chỉnh sửa điểm đo bằng tay.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7311 | Chưa hoàn tất | Chưa xác định |
| Answer Relevancy | 0.7513 | Chưa hoàn tất | Chưa xác định |
| Context Precision | 0.9167 | Chưa hoàn tất | Chưa xác định |
| Context Recall | 0.8250 | Chưa hoàn tất | Chưa xác định |

## Bottom-5 Baseline và đối chiếu Production

Điểm là điểm Baseline, làm tròn bốn chữ số; xếp hạng bằng số chưa làm tròn.

| Hạng | Câu hỏi | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Trung bình |
|---|---|---:|---:|---:|---:|---:|
| 1 | Chu kỳ đổi mật khẩu | 0.0000 | 0.7561 | 0.5000 | 0.5000 | 0.4390 |
| 2 | Bảo hiểm PVI khi thử việc | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |
| 3 | Phạt tạm ứng quá hạn | 0.4000 | 0.7241 | 1.0000 | 0.5000 | 0.6560 |
| 4 | Phê duyệt mua thiết bị 55 triệu | 0.0000 | 0.7511 | 1.0000 | 1.0000 | 0.6878 |
| 5 | Hoàn chi đào tạo 25 triệu | 0.6667 | 0.7112 | 1.0000 | 0.5000 | 0.7195 |

### #1 — Không trả lời được khi có hai phiên bản mật khẩu
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo v2.0 hiện hành, đổi mỗi 120 ngày; quy định 90 ngày của v1.0 đã bị thay thế.
- **Got:** Baseline trả lời “Không tìm thấy thông tin.”
- **Worst metric:** Faithfulness = 0.0000; context precision và recall cùng 0.5000.
- **Bằng chứng context:** Baseline lấy đoạn cũ có mốc 90 ngày, đoạn mới có mốc 120 ngày và đoạn tiêu đề v2.0 riêng. Chu kỳ mới có trong context nhưng bị tách khỏi tiêu đề và thông tin hiệu lực của nó. Production lấy đoạn cha đầy đủ của cả hai bản, có nội dung thay thế bản cũ.
- **Error Tree:** Output đúng? → Không → Context có đáp án? → Có, nhưng lẫn 90/120 ngày và thông tin phiên bản bị tách → Query rõ? → Có → Kiểm tra M1, lọc phiên bản M2 và bước tổng hợp.
- **Root cause:** Quan sát chắc chắn là Baseline không dùng được dữ kiện đã truy xuất. Xung đột phiên bản và mất liên kết giữa điều khoản với tiêu đề là nguyên nhân khả dĩ; không có log suy luận để khẳng định lý do nội bộ khiến mô hình từ chối. Faithfulness thấp ở đây không đồng nghĩa mô hình bịa một con số.
- **Suggested fix:** Giữ nguồn, phiên bản, ngày hiệu lực trên từng chunk; lọc tài liệu bị thay thế khi hỏi quy định hiện hành; dùng đoạn cha và yêu cầu câu trả lời nêu phiên bản nguồn.
- **Đối chiếu Production:** Trả lời “Theo Chính sách mật khẩu (Phiên bản hiện hành) v2.0, mật khẩu phải được thay đổi mỗi 120 ngày.” Đúng nội dung chính khi kiểm tra thủ công. Context vẫn có bản cũ nên việc lọc phiên bản còn cần cải thiện. Chưa có điểm Production để định lượng.

### #2 — Bỏ sót đáp án phủ định về bảo hiểm thử việc
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Không được hưởng PVI; vẫn tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Baseline trả lời “Không tìm thấy thông tin.”
- **Worst metric:** Faithfulness và answer relevancy cùng 0.0000.
- **Bằng chứng context:** Đoạn “Quyền lợi trong thử việc” ghi trực tiếp nhân viên thử việc tham gia bảo hiểm xã hội bắt buộc nhưng chưa được hưởng PVI. Các context khác nói về nhân viên chính thức và gói gia đình. Context recall = 1.0000.
- **Error Tree:** Output đúng? → Không → Context đủ? → Có, có đáp án phủ định trực tiếp → Query rõ đối tượng? → Có → Kiểm tra bước tạo câu trả lời, không mặc định tăng retrieval top-k.
- **Root cause:** Baseline từ chối dù truy xuất được điều khoản cần thiết. Cần cải thiện việc sử dụng điều kiện phủ định và phân biệt nhân viên thử việc/chính thức; chưa đủ bằng chứng để kết luận một lỗi API hay lỗi phân tích ngôn ngữ cụ thể.
- **Suggested fix:** Yêu cầu trả lời Có/Không trước, sau đó nêu điều kiện áp dụng và nguồn. Kiểm tra riêng các câu phủ định. Không xem “Không tìm thấy” là câu trả lời phù hợp khi context chứa điều khoản trực tiếp.
- **Đối chiếu Production:** Trả lời “Không được hưởng bảo hiểm sức khỏe PVI.” Đúng trọng tâm nhưng thiếu thông tin bổ sung về bảo hiểm xã hội trong ground truth. Bổ sung một câu từ `thu_viec.md`; chưa có điểm Production.

### #3 — Giả định tính phí theo ngày và sai số phép tính
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; phí 2%/tháng tương đương 300.000 VNĐ/tháng; nếu pro-rata theo tháng 30 ngày thì khoảng 50.000 VNĐ.
- **Got:** Baseline tính 5 ngày quá hạn và trả lời khoảng 49.950 VNĐ. Production cũng ghi phép tính trung gian khoảng 49.950 VNĐ, rồi kết luận làm tròn 50.000 VNĐ.
- **Worst metric:** Faithfulness = 0.4000; context recall = 0.5000.
- **Bằng chứng context:** `tam_ung.md` ghi hạn 15 ngày và phí 2%/tháng, nhưng không quy định pro-rata hay một tháng 30 ngày. Ground truth có phép quy đổi theo ngày. Baseline còn lấy thêm quy trình hoàn ứng 7 ngày, dễ gây lẫn nghiệp vụ.
- **Error Tree:** Output đúng? → Gần đáp án chuẩn, nhưng phép tính trung gian sai và giả định chưa rõ → Context đủ? → Đủ phí/thời hạn, thiếu quy tắc quy đổi → Query rõ? → Có → Kiểm tra số học, cách nêu giả định và ground truth.
- **Root cause:** Mô hình trình bày giả định 30 ngày/tháng như quy định đã xác nhận. Phép tính chính xác là 15.000.000 × 0,02 × 5/30 = 50.000; nếu dùng 0,1667 thì ra 50.010, không phải 49.950. Điểm thấp có thể liên quan giả định ngoài tài liệu hoặc cách chấm suy luận số học; chưa có verdict từng mệnh đề để xác nhận.
- **Suggested fix:** Tách dữ kiện và giả định: “Phí 300.000 VNĐ/tháng; nếu áp dụng tỷ lệ 5/30 tháng thì 50.000 VNĐ.” Tính bằng công cụ, giữ phân số đến bước cuối. Bổ sung quy tắc pro-rata vào tài liệu hoặc ghi rõ giả định trong đáp án chuẩn. Lọc context theo đúng nghiệp vụ.
- **Đối chiếu Production:** Chỉ lấy chính sách tạm ứng và kết luận cuối khớp ground truth. Tuy nhiên giả định 30 ngày và lỗi phép tính trung gian vẫn còn, nên chưa coi ca này đã được sửa hoàn toàn.

### #4 — Trả lời đúng nhưng faithfulness bằng 0
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Trên 50 triệu cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Baseline trả lời “Đối với đơn hàng trị giá 55 triệu VNĐ (trên 50 triệu VNĐ), người phê duyệt là Tổng Giám đốc (CEO).”
- **Worst metric:** Faithfulness = 0.0000, trong khi context precision và recall đều xấp xỉ/bằng 1.0000.
- **Bằng chứng context:** Bảng `mua_sam.md` ghi “Trên 50.000.000 VNĐ | Tổng Giám đốc (CEO)”. Số tiền 55 triệu nằm trong câu hỏi. Có thêm quy trình hoàn ứng với người duyệt là quản lý trực tiếp, nhưng đáp án đã chọn đúng nghiệp vụ.
- **Error Tree:** Output đúng? → Có, 55 > 50 và đúng CEO → Context hỗ trợ? → Có, bảng thẩm quyền trực tiếp → Query rõ? → Có → Kiểm tra bộ chấm M4 vì điểm bất đồng với bằng chứng.
- **Root cause:** Đây là bất đồng giữa điểm tự động và đánh giá thủ công. Bộ chấm có thể xử lý chưa tốt bảng Markdown hoặc suy luận từ số tiền trong câu hỏi; cần log verdict để biết chính xác. Không có cơ sở gọi câu trả lời này là hallucination chỉ vì faithfulness bằng 0.
- **Suggested fix:** Giữ câu trả lời đúng; kiểm tra verdict khi có hạn mức. Có thể bổ sung biểu diễn văn bản của bảng, trình bày 55 > 50 và nguồn. Dùng kiểm tra thủ công/ngưỡng số học, không sửa điểm RAGAS bằng tay. Lọc context hoàn ứng không cần thiết.
- **Đối chiếu Production:** Trả lời đúng CEO và nêu bảng thẩm quyền mua sắm. Vẫn kèm context hoàn ứng và tạm ứng nên còn cơ hội giảm token, nhiễu. Chưa có điểm Production để biết bộ chấm có cải thiện hay không.

### #5 — Hoàn chi đúng đáp án nhưng recall thấp
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** 8 tháng < 1 năm cam kết; trả 100% chi phí, tức 25.000.000 VNĐ.
- **Got:** Baseline nêu cam kết 1 năm, so sánh 8 < 12 tháng và trả lời hoàn trả toàn bộ 25.000.000 VNĐ.
- **Worst metric:** Context recall = 0.5000; faithfulness = 0.6667.
- **Bằng chứng context:** Đoạn “Cam kết hoàn chi” trong `hoan_chi_dao_tao.md` chứa đủ quy tắc 1 năm và trả 100%. Số tiền 25 triệu, thời gian 8 tháng nằm trong câu hỏi. Một context khác thuộc đào tạo nội bộ với ngân sách 5 triệu/người/năm, không cần cho phép tính này.
- **Error Tree:** Output đúng? → Có → Context đủ quy tắc? → Có → Dữ kiện tình huống ở đâu? → Trong câu hỏi → Kiểm tra M4, nguồn dữ kiện và context không liên quan.
- **Root cause:** Recall thấp không chứng minh thiếu tài liệu. Khả năng bộ chấm đòi con số tình huống trong context hoặc xử lý chưa tốt suy luận từ câu hỏi cần kiểm tra bằng verdict; hiện chưa xác nhận. Có nhiễu giữa đào tạo nội bộ và hoàn chi nhưng câu trả lời không nhầm số tiền.
- **Suggested fix:** Tách ba bước: dữ kiện câu hỏi 25 triệu/8 tháng; quy định nguồn 1 năm/100%; kết quả 25 triệu. Lọc đúng chính sách hoàn chi. Đối chiếu thủ công ca số học và verdict trước khi đổi chunking; không thêm số tiền tình huống vào tài liệu gốc chỉ để tăng điểm.
- **Đối chiếu Production:** Cũng trả lời đúng 25.000.000 VNĐ và giữ đủ chính sách trong đoạn cha. Vẫn có context đào tạo nội bộ; chưa có điểm xác nhận mức cải thiện.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?

**Error Tree walkthrough:**
1. Output đúng? → Baseline nói không tìm thấy, khác đáp án chuẩn.
2. Context đúng? → Có. Điều khoản thử việc trực tiếp nêu chưa được hưởng PVI; recall = 1.0000.
3. Query rewrite OK? → Pipeline không có bước query rewrite. Câu gốc đã rõ đối tượng/quyền lợi; chưa có lý do thêm một lần gọi API.
4. Fix ở bước nào? → Tổng hợp câu trả lời và xử lý điều kiện phủ định. Production giữ đoạn cha đầy đủ và trả lời đúng không có PVI.
5. Kiểm tra sau sửa? → Nêu rõ không có PVI nhưng vẫn tham gia bảo hiểm xã hội bắt buộc; đối chiếu context và chấm Production khi hạn mức cho phép.

Ca này cho thấy retrieval đúng chưa bảo đảm đáp án đúng; tăng top-k hoặc enrichment không phải cách sửa mặc định.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **15 phút:** giữ metadata phiên bản/hiệu lực, thiết kế lọc văn bản đã thay thế.
- **15 phút:** cải thiện prompt trả lời phủ định, kiểm tra điều kiện thử việc/chính thức.
- **15 phút:** kiểm tra phép tính bằng công cụ và nêu rõ giả định pro-rata.
- **15 phút:** giảm context không liên quan, kiểm tra các ca bất đồng giữa RAGAS và đánh giá thủ công; giữ nguyên điểm đã đo.

## Việc còn lại trước khi nộp

1. Tiếp tục chấm Production từ cache, không chạy lại Baseline hay tạo lại đáp án.
2. Khi báo cáo có `evaluation_status: success` và `num_questions: 20`, điền Production và Δ bằng điểm thật.
3. Xếp lại các câu Production theo trung bình bốn metric; cập nhật năm ca theo **bottom-5 Production** đúng yêu cầu. Giữ phân tích Baseline này làm đối chiếu nếu cần.
4. Chạy `python check_lab.py` và kiểm tra thủ công trạng thái báo cáo; script hiện chưa từ chối đầy đủ báo cáo thất bại.
