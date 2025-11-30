import numpy as np
import copy

class BinaryABC:
    def __init__(self, obj_func, n_bits, n_pop=20, max_iter=100, limit=10):
       
        self.obj_func = obj_func
        self.n_bits = n_bits
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.limit = limit
        
        # Khởi tạo quần thể ngẫu nhiên (0 hoặc 1)
        self.population = np.random.randint(0, 2, (n_pop, n_bits))
        self.fitness = np.zeros(n_pop)
        self.trial_counters = np.zeros(n_pop) # Đếm số lần không cải thiện
        
        # Lưu kết quả tốt nhất
        self.best_solution = None
        self.best_fitness = -np.inf
        self.history = []

    def calculate_fitness(self):
        """Tính fitness cho toàn bộ quần thể"""
        for i in range(self.n_pop):
            fit = self.obj_func(self.population[i])
            self.fitness[i] = fit
            
            # Cập nhật Global Best
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_solution = copy.deepcopy(self.population[i])

    def sigmoid(self, x):
        """Hàm chuyển đổi Sigmoid: biến đổi giá trị thực sang khoảng (0, 1)"""
        # np.clip để tránh lỗi tràn số (overflow) khi x quá lớn/nhỏ
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def generate_candidate(self, current_idx):
        """Tạo giải pháp mới dựa trên cơ chế BABC"""
        # Chọn ngẫu nhiên một người hàng xóm khác với hiện tại
        neighbor_idx = current_idx
        while neighbor_idx == current_idx:
            neighbor_idx = np.random.randint(0, self.n_pop)
        
        # 1. Tính vận tốc (giống ABC gốc)
        # phi là số ngẫu nhiên [-1, 1]
        phi = np.random.uniform(-1, 1, self.n_bits)
        
        current_pos = self.population[current_idx]
        neighbor_pos = self.population[neighbor_idx]
        
        # Công thức vận tốc: v = x_i + phi * (x_i - x_k)
        velocity = current_pos + phi * (current_pos - neighbor_pos)
        
        # 2. Chuyển đổi sang xác suất bằng Sigmoid
        probs = self.sigmoid(velocity)
        
        # 3. Tạo giải pháp nhị phân mới
        # Nếu Sigmoid(v) > random() -> bit = 1, ngược lại bit = 0
        rand_vals = np.random.rand(self.n_bits)
        new_solution = (probs > rand_vals).astype(int)
        
        return new_solution

    def run(self):
        # Đánh giá ban đầu
        self.calculate_fitness()
        
        for it in range(self.max_iter):
            # --- GIAI ĐOẠN 1: EMPLOYEED BEES (ONG THỢ) ---
            for i in range(self.n_pop):
                new_sol = self.generate_candidate(i)
                new_fit = self.obj_func(new_sol)
                
                # Cơ chế Greedy Selection (Tham lam)
                if new_fit > self.fitness[i]:
                    self.population[i] = new_sol
                    self.fitness[i] = new_fit
                    self.trial_counters[i] = 0 # Reset bộ đếm
                else:
                    self.trial_counters[i] += 1
            
            # --- GIAI ĐOẠN 2: ONLOOKER BEES (ONG QUAN SÁT) ---
            # Tính xác suất chọn lọc (Roulette Wheel)
            total_fit = np.sum(self.fitness)
            if total_fit == 0:
                probs = np.ones(self.n_pop) / self.n_pop
            else:
                probs = self.fitness / total_fit
                
            # Ong quan sát chọn nguồn thức ăn dựa trên xác suất
            for _ in range(self.n_pop):
                # Chọn chỉ mục (index) dựa trên xác suất
                i = np.random.choice(self.n_pop, p=probs)
                
                new_sol = self.generate_candidate(i)
                new_fit = self.obj_func(new_sol)
                
                if new_fit > self.fitness[i]:
                    self.population[i] = new_sol
                    self.fitness[i] = new_fit
                    self.trial_counters[i] = 0
                else:
                    self.trial_counters[i] += 1
            
            # --- GIAI ĐOẠN 3: SCOUT BEES (ONG TRINH SÁT) ---
            # Tìm ong nào đã vượt quá giới hạn thử (limit)
            for i in range(self.n_pop):
                if self.trial_counters[i] > self.limit:
                    # Tạo lại nguồn thức ăn ngẫu nhiên mới
                    self.population[i] = np.random.randint(0, 2, self.n_bits)
                    self.fitness[i] = self.obj_func(self.population[i])
                    self.trial_counters[i] = 0 # Reset bộ đếm
            
            # Cập nhật Best Global sau mỗi vòng lặp
            current_best_val = np.max(self.fitness)
            if current_best_val > self.best_fitness:
                self.best_fitness = current_best_val
                best_idx = np.argmax(self.fitness)
                self.best_solution = copy.deepcopy(self.population[best_idx])
            
            self.history.append(self.best_fitness)
            print(f"Iteration {it+1}/{self.max_iter}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness

# --- VÍ DỤ SỬ DỤNG ---

# 1. Định nghĩa bài toán: OneMax (Đếm số bit 1)
# Mục tiêu: Tìm chuỗi toàn số 1. Max Fitness = số lượng bit (ở đây là 50)
def objective_function(solution):
    return np.sum(solution)

if __name__ == "__main__":
    # Cấu hình thuật toán
    N_BITS = 50       # Số lượng phần tử (độ dài vector nhị phân)
    N_POP = 20        # Kích thước đàn ong
    MAX_ITER = 50     # Số vòng lặp
    LIMIT = 10        # Giới hạn thử của ong trinh sát

    # Khởi tạo và chạy BABC
    babc = BinaryABC(objective_function, N_BITS, N_POP, MAX_ITER, LIMIT)
    best_sol, best_fit = babc.run()

    print("\n--- KẾT QUẢ ---")
    print(f"Best Fitness: {best_fit}/{N_BITS}")
    print(f"Best Solution: {best_sol}")