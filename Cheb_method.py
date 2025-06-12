import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.interpolate import griddata
import gc

class PoissonSolverApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Решение уравнения Пуассона - Вариант 3")
        self.geometry("1400x900")
        self.V = None
        self.V_exact = None
        self.V_fine = None
        self.history = []

        self.a = -1.0
        self.b = 1.0
        self.c = -1.0
        self.d = 1.0
        self.create_widgets()
        self.set_default_values()

    def set_default_values(self):
        self.n_entry.delete(0, tk.END)
        self.n_entry.insert(0, "20")
        self.m_entry.delete(0, tk.END)
        self.m_entry.insert(0, "20")
        self.max_iter_entry.delete(0, tk.END)
        self.max_iter_entry.insert(0, "50000")
        self.epsilon_entry.delete(0, tk.END)
        self.epsilon_entry.insert(0, "1e-5")

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
        self.params_frame = ttk.LabelFrame(parent, text="Параметры задачи")
        self.params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.n_entry = self.create_entry(self.params_frame, "Число разбиений по x (n):", "20")
        self.m_entry = self.create_entry(self.params_frame, "Число разбиений по y (m):", "20")
        self.a_entry = self.create_entry(self.params_frame, "Левая граница по x (a):", "-1.0")
        self.b_entry = self.create_entry(self.params_frame, "Правая граница по x (b):", "1.0")
        self.c_entry = self.create_entry(self.params_frame, "Нижняя граница по y (c):", "-1.0")
        self.d_entry = self.create_entry(self.params_frame, "Верхняя граница по y (d):", "1.0")
        self.max_iter_entry = self.create_entry(self.params_frame, "Максимальное число итераций:", "50000")
        self.epsilon_entry = self.create_entry(self.params_frame, "Точность метода:", "1e-5")

        self.problem_type_combo = ttk.Combobox(self.params_frame, values=["Тестовая задача", "Основная задача"])
        self.problem_type_combo.current(0)
        ttk.Label(self.params_frame, text="Тип задачи:").pack(pady=5)
        self.problem_type_combo.pack(pady=5)
        
        btn_frame = ttk.Frame(self.params_frame)
        btn_frame.pack(pady=10)

        calc_btn = ttk.Button(btn_frame, text="Рассчитать", command=self.calculate)
        calc_btn.pack(side=tk.LEFT, padx=5)

        self.table_btn = ttk.Button(btn_frame, text="Показать таблицу узлов",
                                    command=self.show_results_table, state=tk.DISABLED)
        self.table_btn.pack(side=tk.LEFT, padx=5)
        
        self.history_btn = ttk.Button(btn_frame, text="Показать историю итераций", 
                                     command=self.show_iteration_history, state=tk.DISABLED)
        self.history_btn.pack(side=tk.LEFT, padx=5)

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
        return 4 * (1 - x ** 2 - y ** 2) * np.exp(1 - x ** 2 - y ** 2)

    def func(self, x, y):
        return abs(x ** 2 - y ** 2)

    def lambda_1(self, h, k):
        return -4 * ((1 / h ** 2) * np.sin(np.pi * h / 2) ** 2 +
                     (1 / k ** 2) * np.sin(np.pi * k / 2) ** 2)

    def lambda_n(self, h, k):
        return -4 * ((1 / h ** 2) * np.cos(np.pi * h / 2) ** 2 +
                     (1 / k ** 2) * np.cos(np.pi * k / 2) ** 2)

    def fill_matrix(self, n, m):
        V = np.zeros((m + 1, n + 1))
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m

        if self.problem_type_combo.get() == "Тестовая задача":
            # Векторизованное заполнение границ
            y_vals = self.c + np.arange(m + 1) * k
            V[:, 0] = self.utest(self.a, y_vals)
            V[:, n] = self.utest(self.b, y_vals)
            
            x_vals = self.a + np.arange(n + 1) * h
            V[0, :] = self.utest(x_vals, self.c)
            V[m, :] = self.utest(x_vals, self.d)
        else:
            # Исправлено: внутренние точки остаются нулевыми
            y_vals = self.c + np.arange(m + 1) * k
            V[:, 0] = -y_vals ** 2 + 1
            V[:, n] = (1 - y_vals ** 2) * np.exp(y_vals)
            
            x_vals = self.a + np.arange(n + 1) * h
            V[0, :] = 1 - x_vals ** 2
            V[m, :] = 1 - x_vals ** 2
        return V

    def residual(self, V, F, h2, k2):
        """Векторизованное вычисление невязки"""
        R = np.zeros_like(V)
        # Внутренние точки
        R[1:-1, 1:-1] = (
            (V[:-2, 1:-1] - 2*V[1:-1, 1:-1] + V[2:, 1:-1]) * h2 +
            (V[1:-1, :-2] - 2*V[1:-1, 1:-1] + V[1:-1, 2:]) * k2 +
            F[1:-1, 1:-1]
        )
        return R

    def interpolate_to_fine_grid(self, V_coarse, n, m):
        """Интерполяция решения на более мелкую сетку"""
        # Создаем координаты для грубой сетки
        x_coarse = np.linspace(self.a, self.b, n + 1)
        y_coarse = np.linspace(self.c, self.d, m + 1)
        
        # Создаем координаты для мелкой сетки
        n_fine = 2 * n
        m_fine = 2 * m
        x_fine = np.linspace(self.a, self.b, n_fine + 1)
        y_fine = np.linspace(self.c, self.d, m_fine + 1)
        
        # Создаем сетку точек для интерполяции
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Преобразуем грубую сетку в точки
        points = []
        values = []
        for i in range(m + 1):
            for j in range(n + 1):
                points.append((x_coarse[j], y_coarse[i]))
                values.append(V_coarse[i, j])
        
        points = np.array(points)
        values = np.array(values)
        
        # Интерполируем на мелкую сетку
        V_fine = griddata(points, values, (X_fine, Y_fine), method='linear')
        return V_fine

    def Cheb(self, F, R, V, n, m, Nmax, eps):
        self.history = [V.copy()]
        
        h = (self.b - self.a) / n
        k_step = (self.d - self.c) / m
        h2 = 1.0 / (h * h)
        k2 = 1.0 / (k_step * k_step)

        lambda1_val = self.lambda_1(h, k_step)
        lambdan_val = self.lambda_n(h, k_step)
        
        lambda_min_A = -lambdan_val
        lambda_max_A = -lambda1_val
        
        mu = lambda_max_A / lambda_min_A
        
        k_chebyshev = 8
        params = []
        for s in range(k_chebyshev):
            denominator = (lambda_min_A + lambda_max_A)/2 + \
                         (lambda_max_A - lambda_min_A)/2 * \
                         np.cos((np.pi * (1 + 2 * s)) / (2 * k_chebyshev))
            tau_s = 1 / denominator
            params.append(tau_s)

        S = 0
        eps_max = float('inf')
        stop_reason = ""
        V_old_block = V.copy()

        while S < Nmax and eps_max > eps:
            V_old_block[:, :] = V
            
            for s_in_block in range(k_chebyshev):
                if S >= Nmax:
                    break
                
                tau_s = params[s_in_block]
                
                # Векторизованное обновление решения
                V[1:-1, 1:-1] += tau_s * R[1:-1, 1:-1]
                
                # Векторизованный пересчет невязки
                R = self.residual(V, F, h2, k2)
                S += 1
                self.history.append(V.copy())

            # Проверка изменения решения (векторизованная)
            delta = V[1:-1, 1:-1] - V_old_block[1:-1, 1:-1]
            eps_max = np.max(np.abs(delta))

        # Определение причины остановки
        if S >= Nmax:
            stop_reason = "Достигнуто максимальное число итераций"
        else:
            stop_reason = "Достигнута заданная точность"

        # Находим точку с максимальной невязкой (векторизованно)
        R_inner = R[1:-1, 1:-1]
        max_abs_idx = np.argmax(np.abs(R_inner))
        i_max, j_max = np.unravel_index(max_abs_idx, R_inner.shape)
        max_res_value = R_inner[i_max, j_max]
        x_point = self.a + (j_max + 1) * h
        y_point = self.c + (i_max + 1) * k_step
        max_res_coords = (x_point, y_point)

        # Оптимизированный расчет невязки без создания временных массивов
        residual_norm = 0.0
        max_res_value = 0.0
        max_i, max_j = 0, 0
        
        for i in range(1, m):
            for j in range(1, n):
                val = R[i, j]
                residual_norm += val ** 2
                if abs(val) > abs(max_res_value):
                    max_res_value = val
                    max_i, max_j = i, j
        
        residual_norm = np.sqrt(residual_norm)
        x_point = self.a + max_j * h
        y_point = self.c + max_i * k_step
        max_res_coords = (x_point, y_point)
        
        if self.problem_type_combo.get() == "Тестовая задача":
            x = np.linspace(self.a, self.b, n + 1)
            y = np.linspace(self.c, self.d, m + 1)
            X, Y = np.meshgrid(x, y)
            exact = self.utest(X, Y)
            error_grid = np.abs(V - exact)
            
            error_inner = error_grid[1:-1, 1:-1]
            max_error = np.max(error_inner)
            max_idx = np.argmax(error_inner)
            i_err, j_err = np.unravel_index(max_idx, error_inner.shape)
            x_err = self.a + (j_err + 1) * h
            y_err = self.c + (i_err + 1) * k_step
            error_coords = (x_err, y_err)
        else:
            max_error = residual_norm
            error_coords = max_res_coords

        return (V, eps_max, max_error, S, residual_norm, max_res_coords, 
                max_res_value, error_coords, lambda1_val, lambdan_val, 
                stop_reason, mu)

    def calculate(self):
        try:
            self.control_frame.pack_forget()
            self.history_btn.config(state=tk.DISABLED)
            self.history = []
            
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

            # Создаем начальное приближение
            V = self.fill_matrix(n, m)
            F = self.matrix_F(n, m)
            
            # Предварительный расчет коэффициентов для residual
            h = (self.b - self.a) / n
            k = (self.d - self.c) / m
            h2 = 1.0 / (h * h)
            k2 = 1.0 / (k * k)
            R = self.residual(V, F, h2, k2)
            
            initial_residual_norm = np.sqrt(np.sum(R[1:-1, 1:-1]**2))
            
            result = self.Cheb(F, R, V, n, m, Nmax, epsilon)
            (V, eps_max, max_error, S, residual_norm, max_res_coords, 
             max_res_value, error_coords, lambda1_val, lambdan_val, 
             stop_reason, mu) = result
            
            x_max, y_max = max_res_coords
            x_error, y_error = error_coords

            self.V = V

            if problem_type == "Тестовая задача":
                x_vals = np.linspace(self.a, self.b, n + 1)
                y_vals = np.linspace(self.c, self.d, m + 1)
                X, Y = np.meshgrid(x_vals, y_vals)
                self.V_exact = self.utest(X, Y)

                result_text = f"""=== РЕЗУЛЬТАТЫ ТЕСТОВОЙ ЗАДАЧИ ===
Параметры расчета:
- Число разбиений: n={n}, m={m}
- Макс. итераций: {Nmax}
- Точность: {epsilon:.1e}
- Число обусловленности: {mu:.4f}
- Начальное приближение: нулевое
- Начальная норма невязки: {initial_residual_norm:.6e}

Результаты:
- Число итераций: {S}
- Причина остановки: {stop_reason}
- Достигнутая точность: {eps_max:.6e}
- СЛАУ решена с нормой невязки: {residual_norm:.6e}
- Максимальная погрешность: {max_error:.6e}
- Точка с макс. погрешностью: x={x_error:.4f}, y={y_error:.4f}"""

                self.results_text.insert(tk.END, result_text)
                self.results_text.config(state=tk.DISABLED)
                
            else:
                self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                coarse_text = f"""=== РЕЗУЛЬТАТЫ ОСНОВНОЙ СЕТКИ ===
Параметры сетки:
- Размер: {n}x{m}
- Макс. итераций: {Nmax}
- Точность: {epsilon:.1e}
- Число обусловленности: {mu:.4f}
- Начальное приближение: нулевое внутри области
- Начальная норма невязки: {initial_residual_norm:.6e}

Результаты:
- Число итераций: {S}
- Причина остановки: {stop_reason}
- Достигнутая точность: {eps_max:.6e}
- СЛАУ решена с нормой невязки: {residual_norm:.6e}
- Максимальная норма невязки: {max_res_value:.6e}
- Точка с макс. погрешностью: x={x_error:.4f}, y={y_error:.4f}"""

                self.results_text.insert(tk.END, coarse_text)
                self.results_text.config(state=tk.DISABLED)
                
                # Сохраняем исходное решение основной сетки
                V_coarse_original = V.copy()
                self.V = V_coarse_original
                
                # Решение на контрольной сетке (без интерполяции)
                n_fine = 2 * n
                m_fine = 2 * m
                
                # Создаем новое начальное приближение для контрольной сетки
                V_fine = self.fill_matrix(n_fine, m_fine)
                F_fine = self.matrix_F(n_fine, m_fine)
                
                # Расчет коэффициентов для мелкой сетки
                h_fine = (self.b - self.a) / n_fine
                k_fine = (self.d - self.c) / m_fine
                h2_fine = 1.0 / (h_fine * h_fine)
                k2_fine = 1.0 / (k_fine * k_fine)
                R_fine = self.residual(V_fine, F_fine, h2_fine, k2_fine)
                
                initial_residual_norm_fine = np.sqrt(np.sum(R_fine[1:-1, 1:-1]**2))
                
                # Увеличиваем число итераций и уменьшаем точность для контрольной сетки
                control_epsilon = epsilon / 10  # Более строгая точность
                control_Nmax = 2 * Nmax  # Увеличили число итераций
                
                fine_result = self.Cheb(F_fine, R_fine, V_fine, n_fine, m_fine, control_Nmax, control_epsilon)
                (V_fine, eps_max_fine, maxeps_fine, S_fine, re_fine, max_res_point_fine, 
                max_res_value_fine, _, lambda1_fine, lambdan_fine, stop_reason_fine, mu_fine) = fine_result
                x_max_fine, y_max_fine = max_res_point_fine
                
                # Сохраняем решение контрольной сетки
                self.V_fine = V_fine
                
                # Взятие каждого второго узла контрольной сетки
                V_fine_coarse = V_fine[::2, ::2]
                
                # Рассчитываем отклонения между исходным решением и прореженным
                deviations = np.abs(V_coarse_original - V_fine_coarse)
                
                # Найдем максимальное отклонение внутри области
                max_deviation = np.max(deviations[1:-1, 1:-1])
                max_idx_dev = np.argmax(deviations[1:-1, 1:-1])
                i_dev, j_dev = np.unravel_index(max_idx_dev, (m-1, n-1))
                x_dev = self.a + (j_dev + 1) * h
                y_dev = self.c + (i_dev + 1) * k
                dev_coords = (x_dev, y_dev)
                
                # Обновим результаты основной сетки
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, coarse_text)
                self.results_text.insert(tk.END, f"\n- Оценка отклонения от контрольной сетки: {max_deviation:.6e}")
                self.results_text.insert(tk.END, f"\n- Точка с макс. отклонением: x={dev_coords[0]:.4f}, y={dev_coords[1]:.4f}")
                self.results_text.config(state=tk.DISABLED)
                
                fine_text = f"""=== РЕЗУЛЬТАТЫ КОНТРОЛЬНОЙ СЕТКИ ===
    Параметры сетки:
    - Размер: {n_fine}x{m_fine}
    - Макс. итераций: {control_Nmax}
    - Точность: {control_epsilon:.1e}
    - Число обусловленности: {mu_fine:.4f}
    - Начальное приближение: нулевое внутри области
    - Начальная норма невязки: {initial_residual_norm_fine:.6e}

    Результаты:
    - Число итераций: {S_fine}
    - Причина остановки: {stop_reason_fine}
    - Достигнутая точность: {eps_max_fine:.6e}
    - СЛАУ решена с нормой невязки: {re_fine:.6e}
    - Максимальная норма невязки: {max_res_value_fine:.6e}
    - Точка с макс. погрешностью: x={x_max_fine:.4f}, y={y_max_fine:.4f}"""

                self.control_text.config(state=tk.NORMAL)
                self.control_text.insert(tk.END, fine_text)
                self.control_text.config(state=tk.DISABLED)
                
                # Сохраняем исходное решение основной сетки
                self.V = V_coarse_original

            self.table_btn.config(state=tk.NORMAL)
            self.history_btn.config(state=tk.NORMAL)

            # Визуализация решения
            self.figure.clear()
            ax = self.figure.add_subplot(111, projection='3d')
            x = np.linspace(self.a, self.b, n + 1)
            y = np.linspace(self.c, self.d, m + 1)
            X, Y = np.meshgrid(x, y)
            Z = np.array(self.V)
            
            # Определяем шаг визуализации в зависимости от размера сетки
            stride_val = max(1, min(n//50, m//50))  # Автоматический подбор шага
            stride_val = max(stride_val, 1)  # Не менее 1
            
            ax.plot_surface(X, Y, Z, cmap='viridis', 
                           rstride=stride_val, cstride=stride_val,
                           alpha=0.8, antialiased=True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('U(x,y)')
            self.canvas.draw()
            
            # Явная очистка больших временных массивов
            del X, Y, Z
            if hasattr(self, 'V_fine'):
                del self.V_fine
            gc.collect()  # Принудительный сбор мусора

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.table_btn.config(state=tk.DISABLED)
            self.history_btn.config(state=tk.DISABLED)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.table_btn.config(state=tk.DISABLED)
            self.history_btn.config(state=tk.DISABLED)

    def matrix_F(self, n, m):
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m
        x = self.a + np.arange(n + 1) * h
        y = self.c + np.arange(m + 1) * k
        X, Y = np.meshgrid(x, y)
        
        if self.problem_type_combo.get() == "Тестовая задача":
            F = self.funcTest(X, Y)
        else:
            F = self.func(X, Y)
        return F

    def show_iteration_history(self):
        if not self.history:
            messagebox.showinfo("История итераций", "История итераций пуста!")
            return
            
        history_window = tk.Toplevel(self)
        history_window.title("История итераций")
        history_window.geometry("1200x800")
        
        # Ограничим количество показываемых итераций
        max_iterations_to_show = min(10, len(self.history))
        
        # Создаем фреймы для каждой итерации
        notebook = ttk.Notebook(history_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for s in range(max_iterations_to_show):
            V_iter = self.history[s]
            n = V_iter.shape[1] - 1
            m = V_iter.shape[0] - 1
            
            # Преобразуем 2D матрицу в 1D вектор
            vector = []
            for i in range(m + 1):
                for j in range(n + 1):
                    vector.append(V_iter[i, j])
            
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=f"Iteration {s}")
            
            frame = ttk.Frame(tab)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            title = ttk.Label(frame, text=f"Приближение v^({s})", font=("Arial", 12, "bold"))
            title.pack(pady=(0, 10))
            
            # Создаем таблицу для вектора
            container = ttk.Frame(frame)
            container.pack(fill=tk.BOTH, expand=True)
            
            tree = ttk.Treeview(container, show="headings", selectmode="extended", columns=("Index", "Value"))
            vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            tree.heading("Index", text="Индекс")
            tree.heading("Value", text="Значение")
            tree.column("Index", width=100, anchor="center")
            tree.column("Value", width=200, anchor="center")
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Заполняем данными
            for idx, value in enumerate(vector):
                tree.insert("", "end", values=(f"{idx}", f"{value:.6e}"))

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
                # Используем сохраненное исходное решение основной сетки
                deviations = np.abs(self.V - self.V_fine[::2, ::2])
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