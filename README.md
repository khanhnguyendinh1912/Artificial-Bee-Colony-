# Artificial-Bee-Colony
kết quả mô phỏng (simulation results)
phần này trình bày cấu hình môi trường và hiệu năng hội tụ của giải thuật abc trong bài toán tối ưu hóa mạng.

thiết lập môi trường (environment setup)

mô phỏng được thực hiện trong không gian 2d kích thước 100m x 100m với các tham số cấu hình:

access points (aps): 10 node (ký hiệu: hình tam giác xanh), bố trí theo cấu trúc lưới (grid layout) để đảm bảo vùng phủ sóng.

user equipment (ue): 4 người dùng (ký hiệu: chấm tròn xanh), phân bố ngẫu nhiên trong không gian.

sensing target: 1 mục tiêu tĩnh (ký hiệu: dấu x đỏ) tại tọa độ cố định (50, 70).

nhiệm vụ hệ thống: đồng thời phục vụ người dùng di động và giám sát mục tiêu tĩnh.

(tham khảo hình ảnh mô hình không gian đính kèm trong thư mục assets)

phân tích hiệu năng (performance analysis)

biểu đồ max-min sinr qua các vòng lặp cho thấy hiệu quả của thuật toán abc:

khả năng hội tụ (convergence): tốc độ hội tụ nhanh, đạt trạng thái bão hòa ổn định chỉ sau khoảng 40 vòng lặp (iterations).

chỉ số sinr:

initial value: -8.25 db

converged value: -6.0 db

kết luận: giải thuật thể hiện khả năng tìm kiếm nghiệm tối ưu cục bộ tốt, cải thiện rõ rệt chất lượng tín hiệu tối thiểu cho người dùng so với mức khởi điểm.
