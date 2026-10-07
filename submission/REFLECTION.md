# Reflection — Lab 21

**Đinh Mạnh Dũng — MSSV 2A202602975**

**1. Kết quả đáng chú ý nhất**

FT đạt target=0,97 so với baseline tối ưu 0,765 nhưng vẫn FAILED vì regression giảm từ 0,7911 xuống 0,6556. Các câu sinh nhật và số tháng cho thấy model sinh JSON thay vì làm theo yêu cầu. Đây là lý do tôi cần xem dữ liệu ngoài miền trước khi quyết định dùng adapter.

**2. Khó khăn trong quá trình thực hiện**

Smoke từng có hai test lỗi sau khi notebook được bổ sung lưu dự đoán. File `.py` đã thay đổi nhưng `.ipynb` dẫn xuất chưa đồng bộ. Thêm bước `scripts/build_colab.py` trước smoke giải quyết vấn đề; lần chạy cuối có 119 tests pass. Tôi cũng xác định `/content` nằm trên máy ảo Colab và tải ZIP về máy để xử lý bằng CLI.

**3. Nhận định cần điều chỉnh về fine-tuning**

Kết quả không cho phép dùng loss thấp hoặc target tăng làm quyết định triển khai duy nhất. Attn_only có loss thấp hơn correct nhưng hòa target; correct học tốt triage nhưng giảm khả năng đáp ứng một số yêu cầu phổ thông. Base được prompt tối ưu là mốc so sánh thiết yếu.

**4. Vai trò của AI assistant và phần phải kiểm chứng**

Tôi dùng AI để đọc repo, chỉnh notebook, chẩn đoán lỗi và hỗ trợ viết report từ artifact. Bản sửa đầu thiếu đồng bộ notebook dẫn xuất, cho thấy kiểm tra cú pháp chưa đủ. Tôi đối chiếu bằng tests, verify và dự đoán thực thay vì lấy số liệu mẫu trong tài liệu.

**5. Bước đầu với bài toán thực tế**

Tôi sẽ xác định phạm vi sử dụng, đóng băng target và eval ngoài miền, rồi đo base với prompt tối ưu trước train. Hướng thử tiếp với adapter này là replay dữ liệu phổ thông, xem gradient overflow và lặp nhiều seed. Tôi chưa giả định replay chắc chắn khắc phục suy giảm.
