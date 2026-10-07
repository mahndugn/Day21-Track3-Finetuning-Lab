# Lab 21 — LoRA cải thiện triage nhưng chưa vượt cổng hồi quy

**Họ tên:** Đinh Mạnh Dũng

**MSSV:** 2A202602975

**Ngày chạy:** 07/10/2026; bắt đầu khoảng 23:03 giờ Việt Nam.

**Ngày hoàn thiện báo cáo:** 08/10/2026.

**Run ID:** `20261007_160332_078654` (thời gian trong ID là UTC).

## 1. Mục tiêu và thiết kế thí nghiệm

Tôi kiểm tra liệu LoRA có giúp model phân loại ticket CSKH tiếng Việt tốt hơn chính base được prompt tối ưu, đồng thời bảo toàn khả năng đáp ứng yêu cầu phổ thông hay không. Tôi chọn `unsloth/Qwen3.5-4B` và dataset mặc định gồm 250 ticket có nhãn `intent`, `urgency`, `product`, `sentiment`. Giữ cấu hình mặc định giúp tập trung kiểm tra mask và đối chứng thay vì thay đồng thời model, dữ liệu và cách đánh giá. Tôi dùng Colab T4 để chạy đầy đủ lab vì GPU laptop của tôi chỉ có 6 GB VRAM.

| Cấu hình | Giá trị thực tế |
|---|---|
| Model | unsloth/Qwen3.5-4B |
| GPU / precision | Tesla T4, tổng VRAM 14,56 GB / fp16 |
| Train / validation | 225 / 25, seed 42 |
| Target / regression eval | 50 / 15, không bật EVAL_LIMIT |
| Mask / max_length | assistant-only / 1024 |
| Epochs / ngân sách steps | 2 / 30 cho cả bốn run |
| Batch mỗi thiết bị / gradient accumulation | 1 / 16; batch hiệu dụng 16 |
| Adapter correct | text-linear, r=16, alpha=32, LR=1e-4, base không lượng tử hóa 4-bit |

NB2 đo baseline trước NB3; log và `stage_timings.json` lưu thứ tự thực hiện. Base model thống nhất cho baseline và fine-tune. Prompt tối ưu (b) giữ nguyên, SHA rút gọn `719e74d3b6232053`. Verify trên Colab xác nhận checksum eval không đổi. Tôi không thay nhãn, prompt hay gate sau khi thấy kết quả. Notebook điều phối chỉ bổ sung lưu dự đoán, log và tạo lại notebook dẫn xuất trước smoke test.

Commit upstream được clone để huấn luyện là `d27c1c02ebe99f32f52f706be88b4c30fb1d7fca`, khác commit notebook điều phối trên fork. Nguồn cấu hình: `results/run_environment.json`, `results/runs.csv`, `results/logs/nb3.log`.

## 2. Bằng chứng mask, template và độ dài

| Kiểm tra | Kết quả |
|---|---|
| answer_is_supervised | true |
| question_is_masked | true |
| Token supervised / tổng token của mẫu proof | 39 / 94 |
| supervised_fraction | 0,4149 |
| Template giữ thẻ mở và nội dung reasoning mẫu kiểm tra | true |

Phần được tính loss khi giải mã ngược:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn này chứa JSON trả lời và token kết thúc, không chứa ticket người dùng. Thẻ đóng `</think>` nằm ở ranh giới phần assistant của template; cả hai assert đều đạt. Chế độ lỗi `everything` tính loss trên 94/94 token, gồm prompt và câu hỏi; tôi không dùng chế độ này để train.

Template check trả `reasoning preserved — safe to train on traces` với mẫu có reasoning thực. Điều này chứng minh trace của mẫu kiểm tra sống sót qua template, không chứng minh adapter cuối cùng giữ năng lực reasoning. Corpus huấn luyện có câu trả lời JSON không kèm trace thực, nên lần chạy này không phải thí nghiệm reasoning-trace collapse.

Thống kê 250 mẫu: mean=93,1; p50=93; p95=98; p99=100; max=101 token; suggested_max_length=256. Tôi giữ 1024 theo tier T4 và dùng nhất quán cho cả bốn run. Mức này lớn hơn nhu cầu đo được, không phải lựa chọn tối ưu theo p95. Mức 256 đã đủ chứa toàn bộ mẫu; đây là hướng tối ưu tài nguyên cho một lần thử tiếp theo. Chi phí còn phụ thuộc cách padding, nên cap 1024 không tự chứng minh chi phí thực tế gấp bốn cap 256.

Nguồn: `results/mask_proof.json`, `results/template_check.json`, `results/token_stats.json`, `results/logs/nb1.log`.

## 3. So sánh ba baseline

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0,0000 | 0,7911 | 0,0000 | 3218,7 |
| (b) Base + optimized prompt | 0,7650 | 0,7911 | 1,0000 | 1022,3 |
| (c) LoRA fine-tune + naive prompt | 0,9700 | 0,6556 | 1,0000 | 1353,5 |

Baseline (b) thực sự mạnh hơn (a): target tăng 76,5 điểm phần trăm, format từ 0 lên 1 và latency giảm khoảng 68,2%. FT tăng thêm 20,5 điểm phần trăm target so với (b), nhưng regression giảm 13,56 điểm phần trăm và latency cao hơn khoảng 32,4%. Prompt ngắn hơn sau fine-tune chưa đồng nghĩa sinh nhanh hơn trong lần đo này.

Target là trung bình tỷ lệ đúng bốn trường, không phải tỷ lệ ticket đúng hoàn toàn. Regression là keyword recall trung bình trên 15 câu, không đại diện toàn bộ kiến thức của model. Format là trung bình tỷ lệ khóa bắt buộc hiện diện trong object trích xuất được; parser có thể lấy JSON trong văn bản bao quanh, nên format=1 không chứng minh tuyệt đối đầu ra chỉ chứa JSON. Latency chưa được lặp nhiều lần để đo độ bất định.

Nguồn: `results/baselines_frozen.json`, `results/verdict.json`. Tôi chấm lại các dự đoán đầy đủ; điểm target và regression khớp các file tổng hợp.

## 4. Đối chứng và diễn biến loss

| Run | Vị trí | r / alpha | Trainable params | LR | Train loss tổng hợp | Target | Format | Train (s) | Peak VRAM (GB) | Latency (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 / 32 | 32.464.896 | 1e-4 | 0,6260 | 0,9700 | 1,0000 | 400,0 | 8,78 | 1353,5 |
| attn_only | q,v | 283 / 566 | 32.456.704 | 1e-4 | 0,5367 | 0,9700 | 1,0000 | 261,5 | 8,79 | 873,6 |
| wrong_lr | text-linear | 16 / 32 | 32.464.896 | 1e-5 | 1,5702 | 0,0000 | 0,0000 | 384,6 | 8,78 | 5110,0 |
| qlora | text-linear, base 4-bit | 16 / 32 | 32.464.896 | 1e-4 | 0,7058 | 0,9400 | 1,0000 | 454,2 | 3,86 | 1731,6 |

Text-linear resolve 12 tên/loại module; q,v resolve 2. Đây không phải số lớp hay tổng số adapter được gắn. Cột `final_loss` trong CSV lấy từ `res.training_loss`, tức loss tổng hợp của run, không phải loss riêng ở step cuối. Cả bốn run ghi max_steps=30; attn_only lệch ngân sách tham số 8.192, khoảng 0,0252%, nhỏ hơn ngưỡng 5%.

**4.1. Vị trí so với rank.** Attn_only dùng matched_rank() để khớp số tham số với correct, thay vì cố định cùng r=16. Hai run hòa target=0,97 dù loss tổng hợp của attn_only thấp hơn. Nếu xếp hạng bằng loss, tôi sẽ gọi attn_only tốt hơn dù thang đo tác vụ không cho kết luận đó. Lần chạy này chưa chứng minh text-linear tốt hơn q,v trên triage; cũng không chứng minh rank cao tự nó gây cải thiện vì rank và vị trí thay đổi cùng nhau dưới ngân sách được kiểm soát. Attn_only nhanh hơn trong cả train và inference, nhưng regression chưa được chấm nên chưa đủ dữ liệu đề xuất triển khai thay correct.

**4.2. Learning rate.** Wrong_lr giữ placement, r, alpha và dữ liệu, chỉ giảm LR danh định 10 lần. Log correct có loss 2,163 tại epoch=0,3556, 0,1396 tại epoch=1 và 0,02578 tại epoch=2. Wrong_lr tương ứng là 2,163; 1,606; 1,119. Loss của wrong_lr có giảm, nên mô tả nó là hoàn toàn phẳng sẽ sai với dữ liệu. Tuy vậy target=0 và format=0 cho thấy cải thiện loss chưa chuyển thành đầu ra đáp ứng tác vụ trong 30 steps. Nếu chỉ thấy loss giảm, tôi có thể kết luận model đã học đủ hoặc cần tăng rank; đối chứng cho thấy nên kiểm tra LR và output thực trước. Kết luận giới hạn ở ngân sách này, không chứng minh LR=1e-5 luôn thất bại nếu train lâu hơn.

**4.3. QLoRA.** Peak VRAM giảm từ 8,78 xuống 3,86 GB, tiết kiệm 4,92 GB, khoảng 56,0%. Đổi lại target giảm 3 điểm phần trăm, train lâu hơn khoảng 13,6% và latency cao hơn khoảng 27,9% so với correct. Format vẫn bằng 1. Kết quả ủng hộ 16-bit khi ưu tiên target và tốc độ, nhưng 4-bit có lợi rõ khi bị giới hạn bộ nhớ. Tôi không khái quát rằng mọi model hay mọi tác vụ đều không nên dùng QLoRA. NB5 chấm adapter này trên base 4-bit đúng cấu hình đã train, tránh sai lệch khi đổi precision trong đánh giá.

Log có grad_norm=nan ở một số mốc, gồm mốc cuối vài run, xen giữa các mốc hữu hạn; loss vẫn hữu hạn. Tôi ghi nhận đây là giới hạn ổn định số học cần xem thêm, không khẳng định mọi optimizer update đều được áp dụng. CSV chứng minh ngân sách steps ghi nhận bằng nhau, chưa có bộ đếm overflow để chứng minh số update thành công bằng nhau. Mỗi cấu hình chỉ có một run, chưa đo biến thiên nhiều seed.

Nguồn: `results/runs.csv`, `results/autopsy.json`, `results/logs/nb3.log`, `results/logs/nb4.log`.

## 5. Phán quyết FAILED do regression

**Target delta = +0,2050; regression delta = −0,1355556; tolerance = 0,02; valid_trace_rate = 0,0.**

Cổng yêu cầu target cao hơn baseline (b) và regression không giảm quá 0,02. Correct đạt điều kiện thứ nhất nhưng không đạt điều kiện thứ hai, nên verdict FAILED được suy ra đúng từ số đo. Sự suy giảm không bị che bởi target=0,97 hay format=1. Dự đoán regression cho thấy model đưa hành vi JSON vào yêu cầu ngoài miền: với yêu cầu viết lời chúc sinh nhật, nó trả object phân loại; với câu hỏi số tháng trong năm, nó trả triage JSON thay vì con số. Đây là bằng chứng suy giảm làm theo yêu cầu, phù hợp với chuyên biệt hóa quá mức. Tuy nhiên keyword recall không phân biệt đầy đủ mất kiến thức với thay đổi hành vi trả lời, nên tôi không khẳng định model quên mọi kiến thức chung. Một số câu vẫn chứa thông tin đúng trong JSON. Tôi giữ gate và chưa đề xuất thay trợ lý đa dụng bằng adapter này. Replay 1–5% dữ liệu phổ thông là hướng thử theo repo, chưa thực hiện và chưa có bằng chứng khắc phục hoàn toàn.

Gate hiện tại chỉ dùng target và regression; format và latency không trực tiếp quyết định PASS/FAIL. Valid_trace_rate=0 cũng không chứng minh reasoning collapse vì corpus không có trace thực và chưa có đối chứng mask trên dữ liệu reasoning.

## 6. Định tính: ca thắng, ca sai và ca thua regression

Target có **33 ca FT thắng, 17 ca hòa và 0 ca FT thua baseline (b)** theo field accuracy. Regression có 1 ca thắng, 9 ca hòa và 5 ca thua theo keyword recall. Tôi dùng ca thua regression để minh họa đánh đổi và ghi rõ nhóm. Nếu rubric đòi riêng hai ca thua trên ticket target, lần chạy này không đáp ứng mục đó; tôi không tạo hoặc sửa nhãn để có ca thua.

Bảng chỉ rút gọn các trường cần so sánh. Đầu ra đầy đủ và nhãn lưu trong `qualitative_comparison.json`, `regression_predictions_b.json` và `regression_predictions_ft.json`. ID bắt đầu từ 0.

| ID / nhóm | Input rút gọn | Nhãn hoặc yêu cầu | Baseline (b) | Fine-tune (c) | Nhận xét |
|---|---|---|---|---|---|
| target 6 | Balo laptop; “Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây.” | intent=doi_tra; urgency=thap; sentiment=tieu_cuc | intent=hoan_tien; urgency=cao; score=0,50 | intent=doi_tra; urgency=thap; score=1,00 | FT thắng: sửa intent và urgency. |
| target 7 | Máy xay sinh tố; “Muốn đổi. Đã 3 ngày rồi. Bực mình.” | intent=doi_tra; urgency=trung_binh | intent=van_chuyen; urgency=cao; score=0,50 | intent=doi_tra; urgency=trung_binh; score=1,00 | FT thắng: không đồng nhất việc chờ 3 ngày với vận chuyển. |
| target 19 | Nồi chiên không dầu; “Vỡ khi nhận. Ngay lập tức. Nhờ shop kiểm tra.” | intent=san_pham_loi; sentiment=trung_tinh | intent=hoan_tien; sentiment=tieu_cuc; score=0,50 | intent=san_pham_loi; sentiment=trung_tinh; score=1,00 | FT thắng: phân biệt sự cố và thái độ. |
| target 3 | Bình giữ nhiệt; “Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.” | urgency=thap | urgency=trung_binh; score=0,75 | urgency=trung_binh; score=0,75 | FT sai nhưng hòa baseline, không gọi đây là FT thua. |
| regression 3 | Viết một câu chúc mừng sinh nhật bằng tiếng Việt | Sinh lời chúc; keyword “sinh nhật” | Có câu chúc; score=1,00 | Object có intent=chuc_mung_sinh_nhat, không có lời chúc; score=0,00 | FT thua rõ cả chỉ số lẫn đáp ứng yêu cầu. |
| regression 9 | Một năm có bao nhiêu tháng? | Keyword “12” | Trả “12 tháng”; score=1,00 | JSON hoi_thong_tin/thap/product=null/trung_tinh; score=0,00 | FT thua: phân loại thay vì trả lời. |
| regression 12 | Ý nghĩa “Có công mài sắt có ngày nên kim” | Keywords kiên trì, cố gắng, bền | score=0,6667 | score=0,3333; vẫn nhắc kiên trì và không nản lòng | Thua theo scorer nhưng ý chính vẫn còn; chưa chứng minh mất hiểu biết. |

Hai ca thua rõ nhất có chung mẫu: áp dụng thói quen JSON cho yêu cầu cần văn bản hoặc câu trả lời trực tiếp. Không phải mọi JSON ngoài miền đều mất điểm: câu hỏi thủ đô vẫn có Hà Nội nên được 1. Regression 6 tăng từ 0 lên 1 khi FT trả 1024, còn baseline hết giới hạn sinh trước khi đưa đáp án. Điều đó nhắc tôi đọc output thay vì chỉ tổng điểm; phép đo chịu ảnh hưởng của giới hạn token và cách tìm từ khóa.

## 7. Kết luận và điều rút ra

Tôi chưa đề xuất triển khai adapter này như một trợ lý đa dụng. Nó cải thiện rõ tác vụ triage, tăng target từ 76,5% lên 97% với đầy đủ khóa theo scorer. Tuy nhiên kết quả đó đi cùng regression giảm 13,56 điểm phần trăm và latency tăng khoảng 32,4% so với base được prompt tối ưu. Việc model trả object phân loại thay lời chúc hoặc câu trả lời về số tháng cho thấy rủi ro nếu người dùng gửi yêu cầu ngoài miền. Nếu ứng dụng chỉ xử lý ticket, kết quả đủ hấp dẫn để thử tiếp trong phạm vi giới hạn, nhưng chưa đủ để bỏ qua gate đã đặt ra. Tôi cần kiểm tra routing, dữ liệu ngoài phân phối và những tình huống nên chuyển yêu cầu sang một model khác.

Đòn bẩy rõ nhất trong đối chứng là thang learning rate: giảm LR 10 lần làm target và format bằng 0 trong cùng ngân sách. Trái lại, text-linear và q,v với số tham số khớp hòa target, nên không thể khẳng định text-linear luôn tốt hơn. QLoRA tiết kiệm bộ nhớ đáng kể nhưng đánh đổi chất lượng và tốc độ. Mask là điều kiện nền tảng đã được chứng minh trước train, không phải biến đã ablate trong bốn run. Ưu tiên tiếp theo là replay dữ liệu phổ thông có kiểm soát, giữ eval và baseline đóng băng, rồi đo lại đủ bốn nhóm. Tôi muốn lặp nhiều seed để phân biệt hiệu ứng cấu hình với biến thiên của một lần chạy. Giá trị của lab là đo được lợi ích và tác hại cùng lúc, thay vì coi loss thấp hoặc target cao là lý do tự động triển khai.

Ba điều tôi rút ra từ bằng chứng:

1. Prompt tối ưu tạo baseline mạnh: base đạt target=0,765 và format=1 mà chưa train. So FT chỉ với prompt đơn giản sẽ phóng đại lợi ích.
2. Loss thấp hơn không bảo đảm target tốt hơn: attn_only có loss=0,5367 so với correct=0,6260, nhưng cùng target=0,97.
3. Target tăng có thể đi cùng suy giảm ngoài miền; dự đoán regression giải thích cụ thể vì sao gate FAILED.

Tôi dùng AI assistant để đọc repo, chỉnh notebook chạy tự động, chẩn đoán lỗi và hỗ trợ tổng hợp report từ artifact. Một lỗi trong bản chỉnh là thay `.py` để lưu dự đoán nhưng chưa đồng bộ `.ipynb`, làm hai test cấu trúc thất bại. Sau khi thêm `scripts/build_colab.py`, lần chạy thực tế có 119 tests pass. Kinh nghiệm này cho thấy review cú pháp chưa đủ; phải kiểm tra nguồn và notebook dẫn xuất. Tôi cũng xác định `/content` nằm trên runtime Colab và tải ZIP về máy để tránh mất artifact.

Nếu có thêm hai giờ, tôi sẽ thử replay tỷ lệ nhỏ được khai báo rõ, giữ ngân sách steps công bằng và lưu thành thí nghiệm riêng. Tôi sẽ xem gradient overflow trước khi khẳng định số update hữu hiệu bằng nhau.

## 8. Tái lập và giới hạn

| Giai đoạn | Giây | Exit code |
|---|---:|---:|
| Smoke | 4,9 | 0 |
| NB1 | 24,7 | 0 |
| NB2 | 391,4 | 0 |
| NB3 | 459,0 | 0 |
| NB4 | 1260,0 | 0 |
| NB5 | 642,2 | 0 |

Tổng core là 2777,3 giây, khoảng 46,29 phút, không gồm Setup, đóng gói và viết report. Đây là số đo của lần chạy, không thay thế thời gian tham khảo chung. NB3 gồm thời gian chuẩn bị; train_seconds=400 chỉ đo phần gọi train.

Stack lưu trong `results/logs/pip_freeze.txt`: Python 3.13.15; torch 2.11.0+cu130; CUDA 13.0; transformers 5.18.0; TRL 1.14.2; PEFT 0.21.1; accelerate 1.15.0; datasets 5.1.0; bitsandbytes 0.50.2; torchao 0.18.0. Verify gốc của Colab có 119 tests pass, chỉ FAIL vì report còn placeholder. `execution_status.json` giữ nguyên trạng thái lịch sử đó, không sửa thành PASS hồi tố. Kiểm tra mới với report đã điền được lưu riêng trong `results/verification_after_report.txt`.

Chọn Option A: report, toàn bộ results, adapter correct và notebook `.py`. ZIP gốc giữ nguyên; `archive_provenance.json` lưu SHA-256. Không đưa HF token hay base weights vào bài nộp. Chưa làm NB6, dataset riêng, đối chứng reasoning-mask, quét rank hoặc HF Hub, nên không yêu cầu điểm thưởng.

Giới hạn: 15 câu regression, scorer từ khóa, một run mỗi cấu hình, chưa đo bất định thống kê hoặc reasoning tổng quát; chưa chấm regression của ba đối chứng. Không có hai ca FT thua trên target, nên định tính khai báo rõ dùng ca thua regression. Chưa có bộ đếm overflow để so số update hữu hiệu. Kết luận chỉ áp dụng cho model, corpus và ngân sách đã ghi nhận.
