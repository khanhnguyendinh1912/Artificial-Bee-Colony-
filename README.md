KẾT QUẢ MÔ PHỎNG (SIMULATION RESULTS)

Phần này trình bày chi tiết về cấu hình môi trường thực nghiệm và phân tích hiệu năng hội tụ của giải thuật, dựa trên cấu hình được định nghĩa trong mã nguồn ABC_Beamforming.py.

Thiết lập Môi trường (Environment Setup)

Môi trường mô phỏng được khởi tạo thông qua lớp SystemEnvironment với không gian 2 chiều kích thước 100m x 100m (được định nghĩa qua biến self.Area_size = 100). Các tham số cấu hình mạng lưới cụ thể như sau:

Access Points (APs): Hệ thống bao gồm 12 điểm truy cập (self.M = 12), mỗi AP được trang bị 4 anten (self.N = 4). Các nút này được bố trí theo cấu trúc lưới đều thông qua hàm initialize_geometry.

User Equipment (UE): Mạng lưới phục vụ 6 người dùng di động (self.K = 6), vị trí được phân bố ngẫu nhiên (biến self.pos_Users).

Sensing Target: Một mục tiêu tĩnh cần giám sát được định nghĩa tại biến self.pos_Target với tọa độ cố định là (50, 70).

Ràng buộc hệ thống:

Công suất phát tối đa: self.P_max_dBm = 23 dBm.

Yêu cầu Sensing SNR tối thiểu: self.Gamma_req_dB = 10 dB.

<img width="1040" height="773" alt="Screenshot from 2026-01-17 23-51-02" src="https://github.com/user-attachments/assets/77564950-bb4d-4c28-a85f-0755d1a0d86f" />


Phân tích Hiệu năng (Performance Analysis)

Hiệu năng của lớp giải thuật ABC_Algorithm được đánh giá qua hàm mục tiêu calculate_obj_fitness. Kết quả trên biểu đồ Max-Min SINR cho thấy:

Khả năng hội tụ (Convergence): Thuật toán được thiết lập chạy trong 200 vòng lặp (self.max_iter = 200). Dữ liệu từ biến history_best_sinr cho thấy hệ thống đạt trạng thái bão hòa ổn định chỉ sau khoảng 40 vòng lặp đầu tiên.

Cải thiện chất lượng tín hiệu:

Initial SINR: Khởi điểm ở mức thấp, xấp xỉ -8.25 dB (tại vòng lặp 0).

Optimized SINR: Sau tối ưu hóa, giá trị Min SINR hội tụ về mức -6.0 dB.

Kết luận: Giải thuật đã tìm được vector beamforming tối ưu (best_w) giúp cân bằng can nhiễu giữa các người dùng, đồng thời thỏa mãn điều kiện phạt (penalty_factor) để đảm bảo ngưỡng Sensing SNR.

<img width="1065" height="773" alt="Screenshot from 2026-01-17 23-48-21" src="https://github.com/user-attachments/assets/8c95ff67-1ebc-48a7-b80e-8c9da75978a7" />

