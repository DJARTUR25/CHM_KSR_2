import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

try:
    from numba import jit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

class PoissonSolverApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Решение уравнения Пуассона - Вариант 3")
        self.geometry("1400x900")
        self.V = None
        self.V_exact = None
        self.V_fine = None

        self.a = -1.0
        self.b = 1.0
        self.c = -1.0
        self.d = 1.0
        self.USE_NUMBA = USE_NUMBA
        self.create_widgets()
        self.set_default_values()

    def set_default_values(self):
        self.n_entry.delete(0, tk.END)
        self.n_entry.insert(0, "650")
        self.m_entry.delete(0, tk.END)
        self.m_entry.insert(0, "650")
        self.max_iter_entry.delete(0, tk.END)
        self.max_iter_entry.insert(0, "5000000")
        self.epsilon_entry.delete(0, tk.END)
        self.epsilon_entry.insert(0, "1e-11")

    def create_widgets(self):
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        center_frame = ttk.Frame(self)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.create_parameters_panel(left_frame)
        self.create_center_panel(center_frame)
        self.create_right_panel(right_frame)

    def create_parameters_panel(self, parent):
        params_frame = ttk.LabelFrame(parent, text="Параметры задачи")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.n_entry = self.create_entry(params_frame, "Число разбиений по x (n):", "100")
        self.m_entry = self.create_entry(params_frame, "Число разбиений по y (m):", "100")
        self.a_entry = self.create_entry(params_frame, "Левая граница по x (a):", "-1.0")
        self.b_entry = self.create_entry(params_frame, "Правая граница по x (b):", "1.0")
        self.c_entry = self.create_entry(params_frame, "Нижняя граница по y (c):", "-1.0")
        self.d_entry = self.create_entry(params_frame, "Верхняя граница по y (d):", "1.0")
        self.max_iter_entry = self.create_entry(params_frame, "Максимальное число итераций:", "10000")
        self.epsilon_entry = self.create_entry(params_frame, "Точность метода:", "1e-10")

        self.problem_type_combo = ttk.Combobox(params_frame, values=["Тестовая задача", "Основная задача"])
        self.problem_type_combo.current(0)
        ttk.Label(params_frame, text="Тип задачи:").pack(pady=5)
        self.problem_type_combo.pack(pady=5)
        
        btn_frame = ttk.Frame(params_frame)
        btn_frame.pack(pady=10)

        calc_btn = ttk.Button(btn_frame, text="Рассчитать", command=self.calculate)
        calc_btn.pack(side=tk.LEFT, padx=5)

        self.table_btn = ttk.Button(btn_frame, text="Показать таблицу узлов",
                                    command=self.show_results_table, state=tk.DISABLED)
        self.table_btn.pack(side=tk.LEFT, padx=5)

    def create_center_panel(self, parent):
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        results_frame = ttk.LabelFrame(parent, text="Результаты основной сетки")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, height=12, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_right_panel(self, parent):
        self.control_frame = ttk.LabelFrame(parent, text="Результаты контрольной сетки")
        self.control_frame.pack(fill=tk.BOTH, expand=True)
        
        self.control_text = tk.Text(self.control_frame, height=12, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(self.control_frame, command=self.control_text.yview)
        self.control_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.control_frame.pack_forget()

    def create_entry(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(frame)
        entry.insert(0, default)
        entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        return entry

    def utest(self, x, y):
        return np.exp(1 - x ** 2 - y ** 2)

    def funcTest(self, x, y):
        return -4 * (1 - x ** 2 - y ** 2) * np.exp(1 - x ** 2 - y ** 2)

    def func(self, x, y):
        return abs(x ** 2 - y ** 2)

    def lambda_1(self, h, k):
        value = -4 * ((1 / h ** 2) * np.sin(np.pi * h / 2) ** 2 +
                     (1 / k ** 2) * np.sin(np.pi * k / 2) ** 2)
        return value

    def lambda_n(self, h, k):
        value = -4 * ((1 / h ** 2) * np.cos(np.pi * h / 2) ** 2 +
                     (1 / k ** 2) * np.cos(np.pi * k / 2) ** 2)
        return value

    def precompute_chebyshev_params(self, k_chebyshev, h, k, n, m):
        lambda1 = self.lambda_1(h, k)
        lambdan = self.lambda_n(h, k)
        
        params = []
        for s in range(k_chebyshev):
            denominator = (lambda1 + lambdan)/2 + (lambdan - lambda1)/2 * np.cos((np.pi * (1 + 2 * s)) / (2 * k_chebyshev))
            tau_s = 1 / denominator
            params.append(tau_s)
        return params

    def fill_matrix(self, n, m):
        V = np.zeros((m + 1, n + 1))
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m

        if self.problem_type_combo.get() == "Тестовая задача":
            for i in range(m + 1):
                y = self.c + i * k
                V[i, 0] = self.utest(self.a, y)
                V[i, n] = self.utest(self.b, y)
            for j in range(n + 1):
                x = self.a + j * h
                V[0, j] = self.utest(x, self.c)
                V[m, j] = self.utest(x, self.d)
        else:
            for i in range(m + 1):
                y = self.c + i * k
                V[i, 0] = -y ** 2 + 1
                V[i, n] = (1 - y ** 2) * np.exp(y)
            for j in range(n + 1):
                x = self.a + j * h
                V[0, j] = 1 - x ** 2
                V[m, j] = 1 - x ** 2

        return V

    def calculate_accuracy(self, V_coarse, V_fine, n, m):
        """Вычисление точности с явным обнулением граничных отклонений"""
        V_fine_subsampled = V_fine[::2, ::2]
        accuracy = np.abs(V_coarse - V_fine_subsampled)
        
        # Явное обнуление граничных узлов
        accuracy[0, :] = 0.0    # Нижняя граница (y = c)
        accuracy[-1, :] = 0.0   # Верхняя граница (y = d)
        accuracy[:, 0] = 0.0    # Левая граница (x = a)
        accuracy[:, -1] = 0.0   # Правая граница (x = b)
        
        return np.max(accuracy)

    def matrix_F(self, n, m):
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m
        F = np.zeros((m + 1, n + 1))
        for i in range(m + 1):
            for j in range(n + 1):
                x = self.a + j * h
                y = self.c + i * k
                if self.problem_type_combo.get() == "Тестовая задача":
                    F[i, j] = self.funcTest(x, y)
                else:
                    F[i, j] = self.func(x, y)
        return F

    def residual(self, V, F, n, m):
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m
        if self.USE_NUMBA:
            return self._residual_numba(V, F, h, k, n, m)
        else:
            return self._residual_original(V, F, n, m)

    def _residual_original(self, V, F, n, m):
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m
        h2 = 1 / h ** 2
        k2 = 1 / k ** 2
        A = -2 * (h2 + k2)

        R = np.zeros((m + 1, n + 1))
        R[1:m, 1:n] = -1 * (
            A * V[1:m, 1:n] +
            h2 * (V[1:m, 2:] + V[1:m, :n-1]) +
            k2 * (V[2:, 1:n] + V[:m-1, 1:n]) -
            F[1:m, 1:n]
        )
        return R

    @staticmethod
    @jit(nopython=True, parallel=True, nogil=True)
    def _residual_numba(V, F, h, k, n, m):
        h2 = 1 / h ** 2
        k2 = 1 / k ** 2
        A = -2 * (h2 + k2)
        R = np.zeros((m + 1, n + 1))
        for i in prange(1, m):
            for j in range(1, n):
                term = A * V[i, j] + \
                       h2 * (V[i, j+1] + V[i, j-1]) + \
                       k2 * (V[i+1, j] + V[i-1, j])
                R[i, j] = -1 * (term - F[i, j])
        return R

    def calculate_residual_norm(self, R, n, m):
        return np.sqrt(np.mean(R[1:m, 1:n] ** 2))
    
    def update_solution(self, V, R, tau, m, n):
        if self.USE_NUMBA:
            self._update_solution_numba(V, R, tau, m, n)
        else:
            V[1:m, 1:n] += tau * R[1:m, 1:n]

    @staticmethod
    @jit(nopython=True, parallel=True, nogil=True)
    def _update_solution_numba(V, R, tau, m, n):
        for i in prange(1, m):
            for j in range(1, n):
                V[i, j] += tau * R[i, j]
    
    def find_max_residual_point(self, R, n, m):
        inner_R = R[1:m, 1:n]
        if inner_R.size == 0:
            return 0.0, (0, 0), (0.0, 0.0)
        
        max_idx = np.unravel_index(np.argmax(np.abs(inner_R)), inner_R.shape)
        max_value = inner_R[max_idx]
        
        i_global = max_idx[0] + 1
        j_global = max_idx[1] + 1
        
        x_point = self.a + j_global * (self.b - self.a) / n
        y_point = self.c + i_global * (self.d - self.c) / m
        
        return max_value, (i_global, j_global), (x_point, y_point)

    def Cheb(self, F, R, V, n, m, Nmax, eps):
        h = (self.b - self.a) / n
        k_step = (self.d - self.c) / m

        lambda1_val = self.lambda_1(h, k_step)
        lambdan_val = self.lambda_n(h, k_step)
        
        # Оптимальный выбор k_chebyshev на основе числа обусловленности
        mu = abs(lambdan_val) / abs(lambda1_val)  # Число обусловленности
        
        # Адаптивный выбор размера блока
        k_chebyshev = 8
        
        # Вычисление параметров Чебышева
        params = []
        for s in range(k_chebyshev):
            denominator = (lambda1_val + lambdan_val)/2 + \
                        (lambdan_val - lambda1_val)/2 * \
                        np.cos((np.pi * (1 + 2 * s)) / (2 * k_chebyshev))
            tau_s = 1 / denominator
            params.append(tau_s)

        S = 0
        eps_max = float('inf')
        stop_reason = ""
        V_old_block = V.copy()  # Для хранения состояния на начало блока

        while S < Nmax and eps_max > eps:
            # Сохраняем текущее состояние в начале блока
            V_old_block[:, :] = V  # Используем view для эффективности
            
            # Выполняем полный блок из k_chebyshev итераций
            for s_in_block in range(k_chebyshev):
                if S >= Nmax:
                    break
                    
                tau_s = params[s_in_block]
                self.update_solution(V, R, tau_s, m, n)
                R = self.residual(V, F, n, m)
                S += 1

            # Вычисляем изменение решения за весь блок
            delta = V[1:m, 1:n] - V_old_block[1:m, 1:n]
            eps_max = np.max(np.abs(delta))

        if S == Nmax and eps_max > eps:
            stop_reason = "Достигнуто максимальное число итераций"
        else:
            stop_reason = "Достигнута заданная точность"

        max_residual_value, max_residual_idx, max_residual_coords = self.find_max_residual_point(R, n, m)
        x_max, y_max = max_residual_coords

        max_error = 0.0
        x_error, y_error = 0.0, 0.0
        
        if self.problem_type_combo.get() == "Тестовая задача":
            x = np.linspace(self.a, self.b, n + 1)
            y = np.linspace(self.c, self.d, m + 1)
            X, Y = np.meshgrid(x, y)
            exact = self.utest(X, Y)
            
            error_grid = np.abs(V - exact)
            max_error = np.max(error_grid)

            # Явно обнуляем погрешность в граничных узлах
            error_grid[0, :] = 0.0    # Нижняя граница
            error_grid[-1, :] = 0.0   # Верхняя граница
            error_grid[:, 0] = 0.0    # Левая граница
            error_grid[:, -1] = 0.0   # Правая граница

            max_error_idx = np.unravel_index(np.argmax(error_grid), error_grid.shape)
            x_error = x[max_error_idx[1]]
            y_error = y[max_error_idx[0]]
        else:
            max_error = self.calculate_residual_norm(R, n, m)
            x_error, y_error = x_max, y_max

        residual_norm = self.calculate_residual_norm(R, n, m)
        
        return V, eps_max, max_error, S, residual_norm, (x_max, y_max), max_residual_value, (x_error, y_error), lambda1_val, lambdan_val, stop_reason

    def calculate(self):
        try:
            self.control_frame.pack_forget()
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.control_text.config(state=tk.NORMAL)
            self.control_text.delete(1.0, tk.END)
            self.control_text.config(state=tk.DISABLED)

            self.a = float(self.a_entry.get())
            self.b = float(self.b_entry.get())
            self.c = float(self.c_entry.get())
            self.d = float(self.d_entry.get())
            n = int(self.n_entry.get())
            m = int(self.m_entry.get())
            Nmax = int(self.max_iter_entry.get())
            epsilon = float(self.epsilon_entry.get())
            problem_type = self.problem_type_combo.get()

            V = np.zeros((m + 1, n + 1))
            S = 0
            converged = True

            if problem_type == "Тестовая задача":
                V = self.fill_matrix(n, m)
                F = self.matrix_F(n, m)
                R = self.residual(V, F, n, m)
                
                initial_residual_norm = self.calculate_residual_norm(R, n, m)
                
                result = self.Cheb(F, R, V, n, m, Nmax, epsilon)
                V, eps_max, maxeps, S, re, max_res_point, max_res_value, error_point, lambda1, lambdan, stop_reason = result
                
                x_max, y_max = max_res_point
                x_error, y_error = error_point
                
                x_vals = np.linspace(self.a, self.b, n + 1)
                y_vals = np.linspace(self.c, self.d, m + 1)
                X, Y = np.meshgrid(x_vals, y_vals)
                self.V_exact = self.utest(X, Y)

                result_text = f"""=== РЕЗУЛЬТАТЫ ТЕСТОВОЙ ЗАДАЧИ ===
Параметры расчета:
- Число разбиений: n={n}, m={m}
- Макс. итераций: {Nmax}
- Точность: {epsilon:.1e}
- Лямбда 1 (λ1): {lambda1:.6e}
- Лямбда n (λn): {lambdan:.6e}
- Число обусловленности: {abs(lambdan/lambda1):.4f}
- Начальное приближение: нулевое
- Начальная норма невязки: {initial_residual_norm:.6e}

Результаты:
- Число итераций: {S}
- Причина остановки: {stop_reason}
- Достигнутая точность: {eps_max:.6e}
- СЛАУ решена с нормой невязки: {re:.6e}
- Максимальная норма невязки: {max_res_value:.6e}
- Точка с макс. нормой невязки: x={x_max:.4f}, y={y_max:.4f}
- Максимальная погрешность: {maxeps:.6e}
- Точка с макс. погрешностью: x={x_error:.4f}, y={y_error:.4f}"""

                self.results_text.insert(tk.END, result_text)
                self.results_text.config(state=tk.DISABLED)
                
            else:
                self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                V_coarse = self.fill_matrix(n, m)
                F_coarse = self.matrix_F(n, m)
                R_coarse = self.residual(V_coarse, F_coarse, n, m)
                
                initial_residual_norm_coarse = self.calculate_residual_norm(R_coarse, n, m)
                
                coarse_result = self.Cheb(F_coarse, R_coarse, V_coarse, n, m, Nmax, epsilon)
                V_coarse, eps_max, maxeps, S, re, max_res_point, max_res_value, _, lambda1_coarse, lambdan_coarse, stop_reason_coarse = coarse_result
                x_max, y_max = max_res_point
                
                coarse_text = f"""=== РЕЗУЛЬТАТЫ ОСНОВНОЙ СЕТКИ ===
Параметры сетки:
- Размер: {n}x{m}
- Макс. итераций: {Nmax}
- Точность: {epsilon:.1e}
- Лямбда 1 (λ1): {lambda1_coarse:.6e}
- Лямбда n (λn): {lambdan_coarse:.6e}
- Число обусловленности: {abs(lambdan_coarse/lambda1_coarse):.4f}
- Начальное приближение: нулевое внутри области
- Начальная норма невязки: {initial_residual_norm_coarse:.6e}

Результаты:
- Число итераций: {S}
- Причина остановки: {stop_reason_coarse}
- Достигнутая точность: {eps_max:.6e}
- СЛАУ решена с Нормой невязки: {re:.6e}
- Максимальная норма невязки: {max_res_value:.6e}
- Точка с макс. нормой невязки: x={x_max:.4f}, y={y_max:.4f}
- Оценка отклонения от контрольной сетки: {maxeps:.6e}"""

                self.results_text.insert(tk.END, coarse_text)
                self.results_text.config(state=tk.DISABLED)
                
                n_fine = 2 * n
                m_fine = 2 * m
                V_fine = self.interpolate_coarse_to_fine(V_coarse, n, m)
                self.V_fine = V_fine 
                F_fine = self.matrix_F(n_fine, m_fine)
                R_fine = self.residual(V_fine, F_fine, n_fine, m_fine)

                initial_residual_norm_fine = self.calculate_residual_norm(R_fine, n_fine, m_fine)

                control_epsilon = epsilon
                control_Nmax = 3 * Nmax

                fine_result = self.Cheb(F_fine, R_fine, V_fine, n_fine, m_fine, control_Nmax, control_epsilon)
                V_fine, eps_max_fine, maxeps_fine, S_fine, re_fine, max_res_point_fine, max_res_value_fine, _, lambda1_fine, lambdan_fine, stop_reason_fine = fine_result
                x_max_fine, y_max_fine = max_res_point_fine

                self.check_boundary_conditions(V_coarse, n, m, "Основная сетка")
                self.check_boundary_conditions(V_fine, n_fine, m_fine, "Контрольная сетка")

                fine_text = f"""=== РЕЗУЛЬТАТЫ КОНТРОЛЬНОЙ СЕТКИ ===
Параметры сетки:
- Размер: {n_fine}x{m_fine}
- Макс. итераций: {control_Nmax}
- Точность: {control_epsilon:.1e}
- Лямбда 1 (λ1): {lambda1_fine:.6e}
- Лямбда n (λn): {lambdan_fine:.6e}
- Число обусловленности: {abs(lambdan_fine/lambda1_fine):.4f}
- Начальное приближение: нулевое
- Начальная норма невязки: {initial_residual_norm_fine:.6e}

Результаты:
- Число итераций: {S_fine}
- Причина остановки: {stop_reason_fine}
- Достигнутая точность: {eps_max_fine:.6e}
- СЛАУ решена с нормой невязки: {re_fine:.6e}
- Максимальная норма невязки: {max_res_value_fine:.6e}
- Точка с макс. нормой невязки: x={x_max_fine:.4f}, y={y_max_fine:.4f}
- Оценка отклонения от контрольной сетки: {maxeps_fine:.6e}"""

                self.control_text.config(state=tk.NORMAL)
                self.control_text.insert(tk.END, fine_text)
                self.control_text.config(state=tk.DISABLED)
                
                V = V_coarse

            self.V = V
            self.table_btn.config(state=tk.NORMAL)

            self.figure.clear()
            ax = self.figure.add_subplot(111, projection='3d')
            x = np.linspace(self.a, self.b, n + 1)
            y = np.linspace(self.c, self.d, m + 1)
            X, Y = np.meshgrid(x, y)
            Z = np.array(V)
            ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('U(x,y)')
            self.canvas.draw()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.table_btn.config(state=tk.DISABLED)

    def interpolate_coarse_to_fine(self, V_coarse, n, m):
        x_coarse = np.linspace(self.a, self.b, n + 1)
        y_coarse = np.linspace(self.c, self.d, m + 1)

        n_fine = 2 * n
        m_fine = 2 * m
        x_fine = np.linspace(self.a, self.b, n_fine + 1)
        y_fine = np.linspace(self.c, self.d, m_fine + 1)

        method_name = "Кубическая"
        
        method_map = {"Кубическая": "cubic", "Линейная": "linear"}
        method = method_map.get(method_name, "cubic")
        interp_func = RegularGridInterpolator((y_coarse, x_coarse), V_coarse, method=method)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        points = np.column_stack((Y_fine.ravel(), X_fine.ravel()))
        V_fine_flat = interp_func(points)
        V_fine = V_fine_flat.reshape((m_fine + 1, n_fine + 1))

        self.apply_boundary_conditions(V_fine, n_fine, m_fine)

        return V_fine
    
    def check_boundary_conditions(self, V, n, m, grid_name):
        """Проверяет выполнение граничных условий"""
        h = (self.b - self.a) / n
        k_step = (self.d - self.c) / m
        errors = []
        
        # Проверка левой границы (x = a)
        for i in range(m + 1):
            y = self.c + i * k_step
            exact = -y**2 + 1
            if abs(V[i, 0] - exact) > 1e-10:
                errors.append(f"Левая граница (y={y:.4f}): {V[i, 0]} ≠ {exact}")
        
        # Проверка правой границы (x = b)
        for i in range(m + 1):
            y = self.c + i * k_step
            exact = (1 - y**2) * np.exp(y)
            if abs(V[i, n] - exact) > 1e-10:
                errors.append(f"Правая граница (y={y:.4f}): {V[i, n]} ≠ {exact}")
        
        # Проверка нижней границы (y = c)
        for j in range(n + 1):
            x = self.a + j * h
            exact = 1 - x**2
            if abs(V[0, j] - exact) > 1e-10:
                errors.append(f"Нижняя граница (x={x:.4f}): {V[0, j]} ≠ {exact}")
        
        # Проверка верхней границы (y = d)
        for j in range(n + 1):
            x = self.a + j * h
            exact = 1 - x**2
            if abs(V[m, j] - exact) > 1e-10:
                errors.append(f"Верхняя граница (x={x:.4f}): {V[m, j]} ≠ {exact}")
        
        if errors:
            print(f"Ошибки граничных условий на {grid_name}:")
            for error in errors[:5]:  # Покажем первые 5 ошибок
                print(error)

    def apply_boundary_conditions(self, V, n, m):
        """Применяет граничные условия к сетке V размерности (m+1) x (n+1)"""
        h = (self.b - self.a) / n
        k_step = (self.d - self.c) / m

        if self.problem_type_combo.get() == "Тестовая задача":
            for i in range(m + 1):
                y = self.c + i * k_step
                V[i, 0] = self.utest(self.a, y)   # левая граница
                V[i, n] = self.utest(self.b, y)   # правая граница
            for j in range(n + 1):
                x = self.a + j * h
                V[0, j] = self.utest(x, self.c)   # нижняя граница
                V[m, j] = self.utest(x, self.d)   # верхняя граница
        else:
            # Основная задача - вариант 3
            for i in range(m + 1):
                y = self.c + i * k_step
                V[i, 0] = -y ** 2 + 1            # левая граница: μ₁(y)
                V[i, n] = (1 - y ** 2) * np.exp(y)  # правая граница: μ₂(y)
            for j in range(n + 1):
                x = self.a + j * h
                V[0, j] = 1 - x ** 2             # нижняя граница: μ₃(x)
                V[m, j] = 1 - x ** 2             # верхняя граница: μ₄(x)

    def richardson_extrapolation(self, V1, V2, n1, m1, n2, m2):
        x2 = np.linspace(self.a, self.b, n2 + 1)
        y2 = np.linspace(self.c, self.d, m2 + 1)
        x1 = np.linspace(self.a, self.b, n1 + 1)
        y1 = np.linspace(self.c, self.d, m1 + 1)
        
        interp_func = RegularGridInterpolator((y2, x2), V2, method='cubic')
        
        X1, Y1 = np.meshgrid(x1, y1)
        points = np.column_stack((Y1.ravel(), X1.ravel()))
        V2_interp_flat = interp_func(points)
        V2_interp = V2_interp_flat.reshape((m1 + 1, n1 + 1))
        
        p = 4
        V_extrapolated = (4**p * V2_interp - V1) / (4**p - 1)
        
        return V_extrapolated

    def show_results_table(self):
        if self.V is None:
            messagebox.showerror("Ошибка", "Сначала выполните расчет!")
            return

        table_window = tk.Toplevel(self)
        table_window.title("Таблицы результатов")
        table_window.geometry("1200x800")

        notebook = ttk.Notebook(table_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        if self.problem_type_combo.get() == "Тестовая задача":
            tab1 = ttk.Frame(notebook)
            notebook.add(tab1, text="Численное решение")
            self.create_table_frame(tab1, self.V, "Численное решение")
            
            tab2 = ttk.Frame(notebook)
            notebook.add(tab2, text="Точное решение")
            self.create_table_frame(tab2, self.V_exact, "Точное решение")
            
            tab3 = ttk.Frame(notebook)
            notebook.add(tab3, text="Погрешность")
            error_grid = np.abs(self.V - self.V_exact)
            self.create_table_frame(tab3, error_grid, "Погрешность решения")

        else:
            tab1 = ttk.Frame(notebook)
            notebook.add(tab1, text="Решение (основная сетка)")
            self.create_table_frame(tab1, self.V, "Решение на основной сетке")
            
            tab2 = ttk.Frame(notebook)
            notebook.add(tab2, text="Отклонения от контрольной сетки")

            if self.V_fine is not None:
                V_fine_coarse = self.V_fine[::2, ::2]
                
                deviations = np.abs(self.V - V_fine_coarse)
                self.create_table_frame(tab2, deviations, "Отклонения от контрольной сетки")
            else:
                label = ttk.Label(tab2, text="Данные контрольной сетки недоступны")
                label.pack(pady=20)

            tab3 = ttk.Frame(notebook)
            notebook.add(tab3, text="Решение (контрольная сетка)")
            if self.V_fine is not None:
                self.create_table_frame(tab3, self.V_fine, "Решение на контрольной сетке")
            else:
                label = ttk.Label(tab3, text="Данные контрольной сетки недоступны")
                label.pack(pady=20)

    def create_table_frame(self, parent, data, title):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = ttk.Label(frame, text=title, font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))

        table_container = ttk.Frame(frame)
        table_container.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(table_container, show="headings", selectmode="extended")

        vsb = ttk.Scrollbar(table_container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        n_cols = data.shape[1]
        tree["columns"] = [f"col{i}" for i in range(n_cols)]

        col_width = max(80, 1000 // n_cols)
        
        for i in range(n_cols):
            x_val = self.a + i * (self.b - self.a) / (n_cols - 1)
            tree.heading(f"col{i}", text=f"x={x_val:.4f}")
            tree.column(f"col{i}", width=col_width, anchor="center")

        n_rows = data.shape[0]
        for j in range(n_rows):
            y_val = self.c + j * (self.d - self.c) / (n_rows - 1)
            values = [f"{val:.6e}" for val in data[j, :]]
            tree.insert("", "end", text=f"y={y_val:.4f}", values=values)

        axis_frame = ttk.Frame(frame)
        axis_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(axis_frame, text="Ось Y →").pack(side=tk.LEFT)
        ttk.Label(axis_frame, text="Ось X →").pack(side=tk.TOP, anchor="e")

if __name__ == "__main__":
    app = PoissonSolverApp()
    app.mainloop()