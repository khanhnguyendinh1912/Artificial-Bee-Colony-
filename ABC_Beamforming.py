import numpy as np
import matplotlib.pyplot as plt
import copy

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & MÔ HÌNH KÊNH TRUYỀN (SYSTEM CONFIGURATION)
# ==============================================================================
class SystemEnvironment:
    def __init__(self):
        # --- Thông số phần cứng ---
        self.M = 12              # Số lượng APs (Access Points)
        self.N = 4              # Số anten trên mỗi AP
        self.K = 6              # Số lượng người dùng (Users)
        
        # --- Công suất & Nhiễu ---
        self.P_max_dBm = 23     # Công suất tối đa mỗi AP (dBm) - Ví dụ: Wifi chuẩn
        self.P_max = 10**((self.P_max_dBm - 30)/10) # Watt
        
        self.Noise_fig_dB = 9   # Hệ số tạp âm
        self.BW = 20e6          # Băng thông 20MHz
        self.Therm_noise = -174 + 10*np.log10(self.BW) + self.Noise_fig_dB # dBm
        self.sigma2 = 10**((self.Therm_noise - 30)/10) # Watt (dùng chung cho Comm/Sens)
        
        # --- Yêu cầu Bài toán ---
        self.Gamma_req_dB = 10  # Yêu cầu Sensing SNR tối thiểu (dB)
        self.Gamma_req = 10**(self.Gamma_req_dB/10)
        
        # --- Hình học không gian (Simulation Area) ---
        self.Area_size = 100    # Khu vực 100m x 100m
        self.PathLoss_Exp = 3.5 # Hệ số suy hao đường truyền

    def initialize_geometry(self):
        """Khởi tạo vị trí AP (lưới đều) và User (ngẫu nhiên)"""
        # 1. Đặt APs theo lưới đều
        sq_M = int(np.sqrt(self.M))
        x_ap = np.linspace(10, self.Area_size-10, int(np.ceil(self.M/sq_M)))
        y_ap = np.linspace(10, self.Area_size-10, sq_M)
        XX, YY = np.meshgrid(x_ap, y_ap)
        self.pos_APs = np.column_stack((XX.ravel()[:self.M], YY.ravel()[:self.M]))
        
        # 2. Đặt Users ngẫu nhiên
        self.pos_Users = np.random.rand(self.K, 2) * self.Area_size
        
        # 3. Đặt Mục tiêu (Target) ở giữa
        self.pos_Target = np.array([self.Area_size/2, self.Area_size/2 + 20])
        
        return self.pos_APs, self.pos_Users, self.pos_Target

    def get_large_scale_fading(self, pos1, pos2):
        """Tính Large-scale fading (Pathloss)"""
        dist = np.linalg.norm(pos1 - pos2)
        dist = max(dist, 1.0) # Tránh chia cho 0
        # Pathloss model: PL(d) = C * d^(-alpha)
        # Simplified Gain:
        PL_dB = -30 - 10 * self.PathLoss_Exp * np.log10(dist)
        return 10**(PL_dB/10)

    def generate_channels(self):
        """Tạo kênh truyền H và ma trận phản xạ A"""
        N_total = self.M * self.N
        
        # 1. Generate Comm Channels H (N_total x K)
        H = np.zeros((N_total, self.K), dtype=complex)
        
        for k in range(self.K):
            for m in range(self.M):
                # Beta: Large scale fading
                beta = self.get_large_scale_fading(self.pos_APs[m], self.pos_Users[k])
                # Small scale fading (Rayleigh)
                h_small = (np.random.randn(self.N, 1) + 1j*np.random.randn(self.N, 1))/np.sqrt(2)
                
                # Gán vào ma trận H tổng
                start_idx = m * self.N
                end_idx = (m+1) * self.N
                H[start_idx:end_idx, k:k+1] = np.sqrt(beta) * h_small

        # 2. Generate Sensing Matrix A (Target Response)
        # A ~ alpha * a(theta) * a(theta)^H
        # Vector dẫn đường a(theta) từ tất cả AP tới Target
        a_vec = np.zeros((N_total, 1), dtype=complex)
        target_rcs = 1.0 + 1j*0.5 # Radar Cross Section giả định
        
        for m in range(self.M):
            beta_tar = self.get_large_scale_fading(self.pos_APs[m], self.pos_Target)
            # Giả sử LoS component dominant cho sensing
            h_los = (np.random.randn(self.N, 1) + 1j*np.random.randn(self.N, 1))/np.sqrt(2)
            
            start_idx = m * self.N
            end_idx = (m+1) * self.N
            a_vec[start_idx:end_idx] = np.sqrt(beta_tar) * h_los

        A_matrix = abs(target_rcs)**2 * np.dot(a_vec, a_vec.conj().T)
        
        return H, A_matrix

# ==============================================================================
# 2. BỘ GIẢI THUẬT ABC (ABC SOLVER CORE)
# ==============================================================================
class ABC_Algorithm:
    def __init__(self, env, H, A):
        self.env = env
        self.H = H
        self.A = A
        
        # --- ABC Hyperparameters ---
        self.n_pop = 10          # Số lượng nguồn thức ăn (Ong)
        self.max_iter = 200      # Số vòng lặp tối đa
        self.limit = 15          # Giới hạn thử nghiệm (cho Scout bee)
        self.penalty_factor = 1e5 # Hệ số phạt cực lớn cho vi phạm Sensing
        
        # Kích thước bài toán
        self.dim = env.M * env.N * env.K # Tổng số biến số phức
        
        # Khởi tạo dữ liệu
        self.population = []     # Chứa các vector w (phẳng)
        self.fitness = np.zeros(self.n_pop)
        self.trials = np.zeros(self.n_pop)
        
        # Lưu lịch sử để vẽ đồ thị
        self.history_best_sinr = []
        self.history_avg_sinr = []

    def repair_power(self, w_flat):
        """
        QUAN TRỌNG: Chuẩn hóa công suất theo từng AP (Per-AP Constraint).
        w_flat: Vector beamforming toàn cục (Flatten)
        """
        # Reshape về ma trận: Dòng = Anten, Cột = User
        N_total = self.env.M * self.env.N
        W_matrix = w_flat.reshape(N_total, self.env.K)
        
        W_repaired = np.copy(W_matrix)
        
        # Duyệt qua từng AP
        for m in range(self.env.M):
            start = m * self.env.N
            end = (m+1) * self.env.N
            
            # Trích xuất trọng số của AP m cho tất cả user
            # Kích thước block: (N_anten, K_users)
            W_AP = W_matrix[start:end, :]
            
            # Công suất phát tổng tại AP m = Trace(W_AP * W_AP^H)
            # = Tổng bình phương độ lớn các phần tử trong block
            p_current = np.sum(np.abs(W_AP)**2)
            
            if p_current > self.env.P_max:
                scale = np.sqrt(self.env.P_max / p_current)
                W_repaired[start:end, :] *= scale
        
        return W_repaired.reshape(-1)

    def calculate_obj_fitness(self, w_flat):
        """
        Trả về:
        1. Min SINR (Giá trị thực tế cần tối ưu)
        2. Fitness (Giá trị dùng để chọn lọc, đã trừ phạt)
        3. Sensing SNR (Để theo dõi)
        """
        N_total = self.env.M * self.env.N
        W = w_flat.reshape(N_total, self.env.K)
        
        # --- 1. Tính Sensing SNR ---
        # w_total = sum(W, axis=1) # Vector tổng hợp phát đi
        # Sensing SNR phụ thuộc vào w_total
        w_total = np.sum(W, axis=1).reshape(-1, 1)
        sens_power = np.real(np.dot(w_total.conj().T, np.dot(self.A, w_total)))
        sens_snr = float(sens_power) / self.env.sigma2
        
        # --- 2. Tính Min SINR ---
        sinrs = []
        for k in range(self.env.K):
            w_k = W[:, k].reshape(-1, 1)
            h_k = self.H[:, k].reshape(-1, 1)
            
            signal = np.abs(np.dot(h_k.conj().T, w_k))**2
            
            # Can nhiễu từ các user khác (j != k)
            interference = 0
            for j in range(self.env.K):
                if j != k:
                    w_j = W[:, j].reshape(-1, 1)
                    interference += np.abs(np.dot(h_k.conj().T, w_j))**2
            
            val = float(signal / (interference + self.env.sigma2))
            sinrs.append(val)
        
        min_sinr = min(sinrs)
        
        # --- 3. Tính Fitness (với hàm phạt) ---
        fitness = min_sinr
        
        # Nếu Sensing không đạt, phạt nặng
        if sens_snr < self.env.Gamma_req:
            violation = self.env.Gamma_req - sens_snr
            # Phạt bậc 2 để càng xa càng bị phạt nặng
            fitness -= self.penalty_factor * (violation**2)
            
        return min_sinr, fitness, sens_snr

    def run(self):
        # --- Giai đoạn Khởi tạo ---
        for i in range(self.n_pop):
            # Tạo ngẫu nhiên phức
            w_init = (np.random.randn(self.dim) + 1j*np.random.randn(self.dim))
            w_init = self.repair_power(w_init) # Đảm bảo ràng buộc công suất
            self.population.append(w_init)
            
            _, fit, _ = self.calculate_obj_fitness(w_init)
            self.fitness[i] = fit

        best_solution_idx = np.argmax(self.fitness)
        best_w = copy.deepcopy(self.population[best_solution_idx])
        global_best_min_sinr, _, _ = self.calculate_obj_fitness(best_w)

        print(f"Bắt đầu tối ưu với {self.max_iter} vòng lặp...")
        
        # --- Vòng lặp chính ---
        for it in range(self.max_iter):
            
            # === 1. EMPLOYED BEES (Ong thợ) ===
            for i in range(self.n_pop):
                # Chọn k ngẫu nhiên khác i
                k = i
                while k == i: k = np.random.randint(self.n_pop)
                
                # Công thức cập nhật ABC: v = x_i + phi * (x_i - x_k)
                phi = np.random.uniform(-1, 1) # Hệ số học hỏi
                v = self.population[i] + phi * (self.population[i] - self.population[k])
                
                # Sửa lỗi công suất
                v = self.repair_power(v)
                
                # Đánh giá tham lam
                _, f_new, _ = self.calculate_obj_fitness(v)
                
                if f_new > self.fitness[i]:
                    self.population[i] = v
                    self.fitness[i] = f_new
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # === 2. ONLOOKER BEES (Ong quan sát) ===
            # Tính xác suất (Roulette Wheel)
            # Xử lý số âm fitness bằng hàm mũ (Softmax style) hoặc tịnh tiến
            fits = self.fitness
            if np.min(fits) < 0:
                fits = fits - np.min(fits) + 1e-5 # Dịch chuyển để tất cả dương
            
            probs = fits / np.sum(fits)
            
            for _ in range(self.n_pop):
                # Chọn i dựa trên xác suất
                i = np.random.choice(range(self.n_pop), p=probs)
                
                k = i
                while k == i: k = np.random.randint(self.n_pop)
                
                phi = np.random.uniform(-1, 1)
                v = self.population[i] + phi * (self.population[i] - self.population[k])
                v = self.repair_power(v)
                
                _, f_new, _ = self.calculate_obj_fitness(v)
                
                if f_new > self.fitness[i]:
                    self.population[i] = v
                    self.fitness[i] = f_new
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # === 3. SCOUT BEES (Ong trinh sát) ===
            for i in range(self.n_pop):
                if self.trials[i] > self.limit:
                    # Reset hoàn toàn
                    w_new = (np.random.randn(self.dim) + 1j*np.random.randn(self.dim))
                    w_new = self.repair_power(w_new)
                    self.population[i] = w_new
                    _, f_new, _ = self.calculate_obj_fitness(w_new)
                    self.fitness[i] = f_new
                    self.trials[i] = 0

            # === TRACKING & LOGGING ===
            # Tìm nghiệm tốt nhất vòng lặp này
            current_best_idx = np.argmax(self.fitness)
            current_best_w = self.population[current_best_idx]
            curr_min_sinr, curr_fit, curr_sens = self.calculate_obj_fitness(current_best_w)
            
            # Cập nhật Global Best
            # Lưu ý: Chỉ cập nhật nếu fitness cao hơn (tức là thỏa mãn tốt hơn cả ràng buộc và mục tiêu)
            if curr_fit > np.max(self.fitness): # Logic đơn giản hóa
                 pass # Đã có trong loop
                 
            # Lưu lại metric thực tế (SINR dB) để vẽ
            self.history_best_sinr.append(10*np.log10(curr_min_sinr + 1e-12))
            
            if (it+1) % 20 == 0:
                print(f"Iter {it+1:3d} | Min SINR: {10*np.log10(curr_min_sinr):.2f} dB | Sensing SNR: {10*np.log10(curr_sens):.2f} dB (Req: {self.env.Gamma_req_dB} dB)")

        return best_w

# ==============================================================================
# 3. CHẠY MÔ PHỎNG VÀ VẼ ĐỒ THỊ
# ==============================================================================
if __name__ == "__main__":
    # 1. Khởi tạo môi trường
    env = SystemEnvironment()
    pos_AP, pos_UE, pos_Tar = env.initialize_geometry()
    
    # 2. Tạo kênh truyền
    print("Đang tạo kênh truyền với suy hao đường truyền (Pathloss)...")
    H_channel, A_target = env.generate_channels()
    
    # 3. Chạy thuật toán ABC
    solver = ABC_Algorithm(env, H_channel, A_target)
    best_w = solver.run()
    
    # 4. Vẽ đồ thị kết quả
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Đồ thị 1: Sự hội tụ của thuật toán
    ax1.plot(solver.history_best_sinr, linewidth=2.5, color='#d62728', label='Max-Min SINR')
    ax1.set_title('Hiệu năng truyền thông qua các vòng lặp', fontsize=14)
    ax1.set_xlabel('Vòng lặp (Iteration)', fontsize=12)
    ax1.set_ylabel('Min SINR của người dùng (dB)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Đồ thị 2: Mô phỏng không gian
    ax2.scatter(pos_AP[:, 0], pos_AP[:, 1], c='blue', marker='^', s=100, label='Access Points')
    ax2.scatter(pos_UE[:, 0], pos_UE[:, 1], c='green', marker='o', s=80, label='Users (UE)')
    ax2.scatter(pos_Tar[0], pos_Tar[1], c='red', marker='x', s=100, linewidth=3, label='Sensing Target')
    ax2.set_title(f'Mô hình không gian ({env.Area_size}m x {env.Area_size}m)', fontsize=14)
    ax2.set_xlabel('Tọa độ X (m)')
    ax2.set_ylabel('Tọa độ Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n--- KẾT QUẢ MÔ PHỎNG ---")
    final_sinr, _, final_sens = solver.calculate_obj_fitness(best_w)
    print(f"1. SINR tối thiểu đạt được (Min SINR): {10*np.log10(final_sinr):.4f} dB")
    print(f"2. SNR Cảm biến đạt được (Sensing SNR): {10*np.log10(final_sens):.4f} dB")
    print(f"3. Trạng thái ràng buộc cảm biến: {'ĐẠT' if final_sens >= env.Gamma_req else 'KHÔNG ĐẠT (Bị phạt)'}")