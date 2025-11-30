import numpy as np
import random
import matplotlib.pyplot as plt


# ==========================================
def objective_function(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

# ==========================================
class ArtificialBeeColony:
    def __init__(self, objective_func, D, SN, MCN, limit, lb, ub):
        self.func = objective_func
        self.D = D  # Số chiều
        self.SN = SN  # Số ong thợ
        self.MCN = MCN  # Số vòng lặp
        self.limit = limit  # Giới hạn thử
        self.lb = lb  # Cận dưới
        self.ub = ub  # Cận trên

        # Khởi tạo mảng dữ liệu
        self.foods = np.zeros((SN, D))
        self.f = np.zeros(SN)
        self.fitness = np.zeros(SN)
        self.trial = np.zeros(SN)
        self.prob = np.zeros(SN)

        # Lưu Global Best
        self.global_best_params = np.zeros(D)
        self.global_best_val = float('inf')

        # Mảng lưu lịch sử để vẽ đồ thị
        self.loss_history = []

    def calculate_fitness(self, f_val):
        if f_val >= 0:
            return 1.0 / (1.0 + f_val)
        else:
            return 1.0 + abs(f_val)

    def initialize(self):
        for i in range(self.SN):
            self.foods[i] = self.lb + np.random.rand(self.D) * (self.ub - self.lb)
            self.f[i] = self.func(self.foods[i])
            self.fitness[i] = self.calculate_fitness(self.f[i])
            self.trial[i] = 0

        # Cập nhật Best ban đầu
        best_idx = np.argmin(self.f)
        self.global_best_val = self.f[best_idx]
        self.global_best_params = self.foods[best_idx].copy()

        # Lưu vào lịch sử
        self.loss_history.append(self.global_best_val)

    # --- HÀM TÌM KIẾM ---
    def _search_neighbor(self, i):
        # Chọn k khác i
        k = list(range(self.SN))
        k.remove(i)
        k = random.choice(k)

        # Chọn chiều j
        j = np.random.randint(self.D)

        # Tham số
        phi = np.random.uniform(-1, 1)

        v_params = self.foods[i].copy()

        v_params[j] = self.foods[i][j] + phi * (self.foods[i][j] - self.foods[k][j])

        # Kiểm tra biên
        v_params[j] = np.clip(v_params[j], self.lb, self.ub)

        # Đánh giá
        f_new = self.func(v_params)
        fit_new = self.calculate_fitness(f_new)

        # Greedy Selection
        if fit_new > self.fitness[i]:
            self.foods[i] = v_params
            self.f[i] = f_new
            self.fitness[i] = fit_new
            self.trial[i] = 0

            # Cập nhật ngay Global Best nếu tốt hơn
            if f_new < self.global_best_val:
                self.global_best_val = f_new
                self.global_best_params = v_params.copy()
        else:
            self.trial[i] += 1

    def run(self):
        self.initialize()

        for cycle in range(self.MCN):
            # 1. Ong thợ
            for i in range(self.SN):
                self._search_neighbor(i)

            # 2. Tính xác suất
            sum_fit = np.sum(self.fitness)
            if sum_fit == 0:
                self.prob = np.ones(self.SN) / self.SN
            else:
                self.prob = self.fitness / sum_fit

            # 3. Ong quan sát
            t = 0
            i = 0
            while t < self.SN:
                if np.random.rand() < self.prob[i]:
                    t += 1
                    self._search_neighbor(i)
                i = (i + 1) % self.SN

            # 4. Ong trinh sát
            max_trial_idx = np.argmax(self.trial)
            if self.trial[max_trial_idx] > self.limit:
                self.foods[max_trial_idx] = self.lb + np.random.rand(self.D) * (self.ub - self.lb)
                self.f[max_trial_idx] = self.func(self.foods[max_trial_idx])
                self.fitness[max_trial_idx] = self.calculate_fitness(self.f[max_trial_idx])
                self.trial[max_trial_idx] = 0

            # Cập nhật Global Best lần cuối trong vòng
            current_best_idx = np.argmin(self.f)
            if self.f[current_best_idx] < self.global_best_val:
                self.global_best_val = self.f[current_best_idx]
                self.global_best_params = self.foods[current_best_idx].copy()

            # Lưu lịch sử để vẽ
            self.loss_history.append(self.global_best_val)

            if (cycle + 1) % 10 == 0:
                print(f"Cycle {cycle + 1}: Best f(x) = {self.global_best_val:.10f}")

        return self.global_best_val, self.global_best_params


# ==========================================
if __name__ == "__main__":
    # Cấu hình tham số
    D = 5
    SN = 20
    MCN = 100
    limit = 10
    lb = -5
    ub = 5

    print("--- BẮT ĐẦU CHẠY STANDARD ABC ---")
    abc = ArtificialBeeColony(objective_function, D, SN, MCN, limit, lb, ub)
    best_val, best_params = abc.run()

    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Giá trị nhỏ nhất (Min): {best_val}")
    print(f"Tại vị trí (x): {best_params}")

    # --- VẼ ĐỒ THỊ ---
    plt.figure(figsize=(10, 6))
    plt.plot(abc.loss_history, color='blue', linewidth=2, marker='o', markersize=3)

    plt.title('Biểu đồ hội tụ của thuật toán Standard ABC')
    plt.xlabel('Vòng lặp (Iteration)')
    plt.ylabel('Giá trị hàm mục tiêu f(x)')
    plt.grid(True)
    plt.yscale('log')
    plt.ylim(1e-4, 100)
    plt.legend()

    plt.show()