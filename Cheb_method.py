import scipy
from scipy.interpolate import interp2d
import sys
import math
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class PoissonSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Решение уравнения Пуассона - Вариант 3")
        self.geometry("1200x900")
        self.a = -1.0
        self.b = 1.0
        self.c = -1.0
        self.d = 1.0
        self.create_widgets()
        self.set_default_values()

    def set_default_values(self):
        self.n_entry.delete(0, tk.END)
        self.n_entry.insert(0, "100")
        self.m_entry.delete(0, tk.END)
        self.m_entry.insert(0, "100")
        self.max_iter_entry.delete(0, tk.END)
        self.max_iter_entry.insert(0, "5000")
        self.epsilon_entry.delete(0, tk.END)
        self.epsilon_entry.insert(0, "1e-8")

    def create_widgets(self):
        # Основные фреймы
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Левая панель - параметры
        self.create_parameters_panel(left_frame)

        # Правая панель - график и результаты
        self.create_results_panel(right_frame)

    def create_parameters_panel(self, parent):
        params_frame = ttk.LabelFrame(parent, text="Параметры задачи")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Поля ввода
        self.n_entry = self.create_entry(params_frame, "Число разбиений по x (n):", "20")
        self.m_entry = self.create_entry(params_frame, "Число разбиений по y (m):", "20")
        self.a_entry = self.create_entry(params_frame, "Левая граница по x (a):", "-1.0")
        self.b_entry = self.create_entry(params_frame, "Правая граница по x (b):", "1.0")
        self.c_entry = self.create_entry(params_frame, "Нижняя граница по y (c):", "-1.0")
        self.d_entry = self.create_entry(params_frame, "Верхняя граница по y (d):", "1.0")
        self.max_iter_entry = self.create_entry(params_frame, "Максимальное число итераций:", "1000")
        self.epsilon_entry = self.create_entry(params_frame, "Точность метода:", "0.0000005")

        # Выбор типа задачи
        self.problem_type_combo = ttk.Combobox(params_frame, values=["Тестовая задача", "Основная задача"])
        self.problem_type_combo.current(0)
        ttk.Label(params_frame, text="Тип задачи:").pack(pady=5)
        self.problem_type_combo.pack(pady=5)

        # Кнопки
        btn_frame = ttk.Frame(params_frame)
        btn_frame.pack(pady=10)

        calc_btn = ttk.Button(btn_frame, text="Рассчитать", command=self.calculate)
        calc_btn.pack(side=tk.LEFT, padx=5)

        self.table_btn = ttk.Button(btn_frame, text="Показать таблицу узлов",
                                    command=self.show_results_table, state=tk.DISABLED)
        self.table_btn.pack(side=tk.LEFT, padx=5)

    def create_results_panel(self, parent):
        # График
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Текстовые результаты
        self.results_text = tk.Text(parent, height=10, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

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

    def lambda_1(self, h, k, n, m):
        return -4 * ((1 / h ** 2) * np.sin(np.pi / (2 * n)) ** 2 +
                     (1 / k ** 2) * np.sin(np.pi / (2 * m)) ** 2)

    def lambda_n(self, h, k, n, m):
        return -4 * ((1 / h ** 2) * np.cos(np.pi / (2 * n)) ** 2 +
                     (1 / k ** 2) * np.cos(np.pi / (2 * m)) ** 2)

    def precompute_chebyshev_params(self, k_chebyshev, h, k, n, m):
        """Предварительно вычисляет параметры Чебышева"""
        lambda1 = self.lambda_1(h, k, n, m)
        lambdan = self.lambda_n(h, k, n, m)

        params = []
        for s in range(k_chebyshev):
            tau_s = 2.0 / (
                    lambda1 + lambdan +
                    (lambdan - lambda1) *
                    np.cos((np.pi * (2 * s + 1)) / (2 * k_chebyshev))
            )
            params.append(tau_s)
        return params

    def fill_matrix(self, n, m):
        V = np.zeros((m+1, n+1))
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m

        # Сначала заполняем границы по y (нижняя и верхняя)
        for j in range(n+1):
            x = self.a + j * h
            V[0, j] = 1 - x**2  # μ₃(x) = 1 - x²
            V[m, j] = 1 - x**2  # μ₄(x) = 1 - x²

        # Затем заполняем границы по x (левая и правая)
        for i in range(m+1):
            y = self.c + i * k
            V[i, 0] = -y**2 + 1  # μ₁(y) = -y² + 1
            V[i, n] = (1 - y**2) * np.exp(y)  # μ₂(y) = (1 - y²)e^y

        return V

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
        h2 = 1 / h ** 2
        k2 = 1 / k ** 2
        A = -2 * (h2 + k2)

        R = np.zeros((m + 1, n + 1))

        # Векторизованное вычисление невязки (исправлен знак перед F)
        R[1:m, 1:n] = (A * V[1:m, 1:n] +
                       h2 * (V[1:m, 2:n + 1] + V[1:m, 0:n - 1]) +
                       k2 * (V[2:m + 1, 1:n] + V[0:m - 1, 1:n]) -
                       F[1:m, 1:n])  # Знак минус перед F - исправлено!
        return R

    def calculate_residual_norm(self, R, n, m):
        """Вычисление нормы невязки"""
        return np.sqrt(np.mean(R[1:m, 1:n] ** 2))

    def Cheb(self, F, R, V, n, m, Nmax, eps):
        h = (self.b - self.a) / n
        k = (self.d - self.c) / m

        # Оптимальное число параметров Чебышева
        mu = abs(self.lambda_n(h, k, n, m) / self.lambda_1(h, k, n, m))
        k_chebyshev = min(50, max(10, int(np.log(mu + 1))))
        print(k_chebyshev)

        # Предварительное вычисление параметров Чебышева
        cheb_params = self.precompute_chebyshev_params(k_chebyshev, h, k, n, m)

        S = 0
        eps_max = 0
        stop_reason = "Достигнуто максимальное число итераций"

        # Основной цикл итераций с правильной остановкой
        while S < Nmax:
            # Выбор параметра Чебышева для текущей итерации
            tau_s = cheb_params[S % k_chebyshev]

            # Векторизованное обновление решения
            V_old = V.copy()
            V[1:-1, 1:-1] -= tau_s * R[1:m, 1:n]

            # Вычисление изменения и невязки
            eps_max = np.max(np.abs(V[1:m, 1:n] - V_old[1:m, 1:n]))
            R = self.residual(V, F, n, m)

            S += 1
            if eps_max < eps:
                stop_reason = f"Достигнута заданная точность: {eps_max:.6e} < {eps}"
                break
            if S >= Nmax:  # Явная проверка достижения лимита итераций
                break

        # Вычисление ошибки
        if self.problem_type_combo.get() == "Тестовая задача":
            # Для тестовой задачи сравниваем с точным решением
            x = np.linspace(self.a, self.b, n + 1)
            y = np.linspace(self.c, self.d, m + 1)
            X, Y = np.meshgrid(x, y)
            exact = self.utest(X, Y)
            max_error = np.max(np.abs(V - exact))
            error_point = np.unravel_index(np.argmax(np.abs(V - exact)), V.shape)
        else:
            # Для основной задачи используем норму невязки
            max_error = self.calculate_residual_norm(R, n, m)
            error_point = (0, 0)

        residual_norm = self.calculate_residual_norm(R, n, m)
        return V, eps_max, max_error, S, residual_norm, error_point, stop_reason

    def interpolate_coarse_to_fine(self, V_coarse, n, m):
        """Билинейная интерполяция с грубой сетки (n, m) на (2n, 2m)."""
        x_coarse = np.linspace(self.a, self.b, n + 1)
        y_coarse = np.linspace(self.c, self.d, m + 1)

        # Создаем интерполятор (обратите внимание на порядок осей: (y, x))
        interp_func = RegularGridInterpolator((y_coarse, x_coarse), V_coarse)

        # Новая сетка
        n_fine = 2 * n
        m_fine = 2 * m
        x_fine = np.linspace(self.a, self.b, n_fine + 1)
        y_fine = np.linspace(self.c, self.d, m_fine + 1)

        # Создаем координаты для интерполяции в виде списка точек (y, x)
        points = np.array([[y, x] for y in y_fine for x in x_fine])

        # Интерполяция
        V_fine_flat = interp_func(points)
        V_fine = V_fine_flat.reshape((m_fine + 1, n_fine + 1))

        return V_fine

    def calculate(self):
        try:
            # Получение параметров
            self.a = float(self.a_entry.get())
            self.b = float(self.b_entry.get())
            self.c = float(self.c_entry.get())
            self.d = float(self.d_entry.get())
            n = int(self.n_entry.get())
            m = int(self.m_entry.get())
            Nmax = int(self.max_iter_entry.get())
            epsilon = float(self.epsilon_entry.get())
            problem_type = self.problem_type_combo.get()

            # Объявление переменных
            V = np.zeros((m + 1, n + 1))
            S = 0
            epsilon2 = 0.0
            stop_reason = ""
            method_info = ""

            # Выполнение расчета
            if problem_type == "Тестовая задача":
                method_info = "Чебышевский итерационный метод"
                # Начальное приближение - границы из точного решения, внутри нули
                h = (self.b - self.a) / n
                k = (self.d - self.c) / m
                
                # Заполняем границы
                for j in range(n + 1):
                    x = self.a + j * h
                    V[0, j] = self.utest(x, self.c)  # нижняя граница
                    V[m, j] = self.utest(x, self.d)  # верхняя граница
                    
                for i in range(m + 1):
                    y = self.c + i * k
                    V[i, 0] = self.utest(self.a, y)  # левая граница
                    V[i, n] = self.utest(self.b, y)  # правая граница

                F = self.matrix_F(n, m)
                R = self.residual(V, F, n, m)
                V, eps_max, maxeps, S, re, punto, stop_reason = self.Cheb(F, R, V, n, m, Nmax, epsilon)
            else:
                method_info = "Чебышевский итерационный метод с контрольной сеткой"
                # Решение на основной сетке
                V_coarse = self.fill_matrix(n, m)
                F_coarse = self.matrix_F(n, m)
                R_coarse = self.residual(V_coarse, F_coarse, n, m)
                V_coarse, eps_max, maxeps, S, re, _, stop_reason = self.Cheb(F_coarse, R_coarse, V_coarse, n, m, Nmax, epsilon)

                # Решение на контрольной сетке с повышенной точностью
                n_fine = 2 * n
                m_fine = 2 * m
                V_fine = self.interpolate_coarse_to_fine(V_coarse, n, m)
                F_fine = self.matrix_F(n_fine, m_fine)
                R_fine = self.residual(V_fine, F_fine, n_fine, m_fine)
                V_fine, _, _, _, _, _, _ = self.Cheb(F_fine, R_fine, V_fine, n_fine, m_fine, 2 * Nmax, epsilon / 10)

                # Нормированное отклонение
                h_coarse = (self.b - self.a) / n
                x_coarse = np.linspace(self.a, self.b, n + 1)
                y_coarse = np.linspace(self.c, self.d, m + 1)

                # Находим соответствующие точки на мелкой сетке
                x_fine_idx = np.arange(0, 2 * n + 1, 2)
                y_fine_idx = np.arange(0, 2 * m + 1, 2)

                # Вычисляем максимальное отклонение
                epsilon2 = np.max(np.abs(V_coarse - V_fine[y_fine_idx[:, None], x_fine_idx])) / np.max(np.abs(V_coarse))

                V = V_coarse

            self.V = V
            self.table_btn.config(state=tk.NORMAL)

            # Формирование отчета
            result_text = f"""=== РЕЗУЛЬТАТЫ ===
Параметры расчета:
- Тип задачи: {problem_type}
- Число разбиений: n={n}, m={m}
- Макс. итераций: {Nmax}
- Точность: {epsilon}
- Метод решения: {method_info}

Результаты:
- Причина остановки: {stop_reason}
- Число выполненных итераций: {S}
- Максимальное изменение на последней итерации: {eps_max:.6e}
- Норма невязки на последней итерации: {re:.6e}
- Максимальная погрешность: {maxeps:.6e}"""

            if problem_type == "Основная задача":
                result_text += f"\n- Отклонение от контрольной сетки: {epsilon2:.6e}"

            # Критерии остановки
            result_text += f"""
            
=== КРИТЕРИИ ОСТАНОВКИ ===
- По количеству итераций: {S} из {Nmax} ({(S/Nmax*100):.1f}%)
- По точности: заданная {epsilon}, достигнутая {eps_max:.6e}"""

            # Обновление текстового поля
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            self.results_text.config(state=tk.DISABLED)

            # Обновление графика
            self.figure.clear()
            ax = self.figure.add_subplot(111, projection='3d')
            x = np.linspace(self.a, self.b, n + 1)
            y = np.linspace(self.c, self.d, m + 1)
            X, Y = np.meshgrid(x, y)
            Z = np.array(V)
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('U(x,y)')
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.table_btn.config(state=tk.DISABLED)

    def show_results_table(self):
        if self.V is None:
            messagebox.showerror("Ошибка", "Сначала выполните расчет!")
            return

        # Создаем новое окно
        table_window = tk.Toplevel(self)
        table_window.title("Таблица решений в узлах сетки")
        table_window.geometry("800x600")

        # Создаем Treeview с прокруткой
        frame = ttk.Frame(table_window)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, show="headings")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Настраиваем колонки
        n = self.V.shape[1]
        tree["columns"] = [f"x{i}" for i in range(n)]
        for i in range(n):
            tree.column(f"x{i}", width=100, anchor="center")
            tree.heading(f"x{i}", text=f"X={self.a + i * (self.b - self.a) / (n - 1):.2f}")

        # Изменение 1: Перевернем порядок строк (от максимального Y к минимальному)
        # Вычисляем количество строк
        num_rows = self.V.shape[0]
        
        # Изменение 2: Добавляем данные в обратном порядке
        for row_idx in range(num_rows-1, -1, -1):
            # Вычисляем значение Y (от d к c)
            y_val = self.d - row_idx * (self.d - self.c) / (num_rows - 1)
            
            values = [f"{val:.6e}" for val in self.V[row_idx, :]]
            tree.insert("", "end", text=f"Y={y_val:.2f}", values=values)

        # Добавляем подписи осей
        ttk.Label(table_window, text="Ось Y →").pack(side=tk.LEFT, anchor="nw")
        ttk.Label(table_window, text="Ось X →").pack(side=tk.TOP, anchor="ne")


if __name__ == "__main__":
    app = PoissonSolverApp()
    app.mainloop()