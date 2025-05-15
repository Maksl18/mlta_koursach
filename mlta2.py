import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import re
import time
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from collections import defaultdict, deque
from scipy.spatial import ConvexHull
import matplotlib.patches as patches


class Point:
    """Клас для представлення точки на площині"""
    def __init__(self, x, y, id=None):
        self.x = x
        self.y = y
        self.id = id
    
    def __str__(self):
        return f"Point {self.id}: x={self.x}, y={self.y}"
    
    def distance_to(self, other_point):
        """Обчислює відстань між двома точками"""
        return ((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2) ** 0.5


class Polygon:
    """Клас для представлення багатокутника"""
    def __init__(self, points=None, id=None):
        self.points = points if points else []
        self.id = id
        self.area = self._calculate_area() if points else 0
        
    def add_point(self, point):
        """Додає точку до багатокутника"""
        self.points.append(point)
        self.area = self._calculate_area()
        
    def _calculate_area(self):
        """Обчислює площу багатокутника за формулою Гаусса"""
        if len(self.points) < 3:
            return 0
            
        area = 0
        n = len(self.points)
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x * self.points[j].y
            area -= self.points[j].x * self.points[i].y
        return abs(area) / 2
        
    def is_point_inside(self, point):
        """Перевіряє, чи знаходиться точка всередині багатокутника
        використовуючи алгоритм з підрахунком перетинів променя"""
        if not self.points:
            return False
        
        n = len(self.points)
        inside = False
        
        p1x, p1y = self.points[0].x, self.points[0].y
        for i in range(n + 1):
            p2x, p2y = self.points[i % n].x, self.points[i % n].y
            if point.y > min(p1y, p2y):
                if point.y <= max(p1y, p2y):
                    if point.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def contains_polygon(self, other_polygon):
        """Перевіряє, чи містить даний багатокутник інший багатокутник повністю"""
        return all(self.is_point_inside(point) for point in other_polygon.points)
    
    def intersects_with(self, other_polygon):
        """Перевіряє, чи перетинається даний багатокутник з іншим"""
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            
            for j in range(len(other_polygon.points)):
                p3 = other_polygon.points[j]
                p4 = other_polygon.points[(j + 1) % len(other_polygon.points)]
                
                if self._do_line_segments_intersect(p1, p2, p3, p4):
                    return True
        
        return False
    
    def _do_line_segments_intersect(self, p1, p2, p3, p4):
        """Перевіряє, чи перетинаються два відрізки"""
        def direction(a, b, c):
            return (c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y)
        
        def on_segment(p, q, r):
            return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                    q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))
        
        d1 = direction(p3, p4, p1)
        d2 = direction(p3, p4, p2)
        d3 = direction(p1, p2, p3)
        d4 = direction(p1, p2, p4)
        
        # Перетин сегментів
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        
        # Спеціальні випадки
        if d1 == 0 and on_segment(p3, p1, p4):
            return True
        if d2 == 0 and on_segment(p3, p2, p4):
            return True
        if d3 == 0 and on_segment(p1, p3, p2):
            return True
        if d4 == 0 and on_segment(p1, p4, p2):
            return True
        
        return False
    
    def get_x_coords(self):
        """Повертає список x-координат всіх точок багатокутника"""
        return [point.x for point in self.points]
    
    def get_y_coords(self):
        """Повертає список y-координат всіх точок багатокутника"""
        return [point.y for point in self.points]

    def get_center(self):
        """Повертає центр багатокутника"""
        if not self.points:
            return None
        x_sum = sum(point.x for point in self.points)
        y_sum = sum(point.y for point in self.points)
        return Point(x_sum / len(self.points), y_sum / len(self.points))


class PolygonProcessor:
    """Базовий клас для обробки багатокутників"""
    def __init__(self):
        self.all_points = []
        self.polygons = []
        self.nested_structure = {}
        
    def load_points_from_file(self, file_path):
        """Завантажує точки з текстового файлу"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Використання регулярного виразу для пошуку координат
            pattern = r'Point\s+(\d+):\s+x=(-?\d+(?:\.\d+)?),\s+y=(-?\d+(?:\.\d+)?)'
            matches = re.findall(pattern, content)
            
            self.all_points = []
            for match in matches:
                point_id, x, y = match
                self.all_points.append(Point(float(x), float(y), int(point_id)))
                
            return True
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при зчитуванні файлу: {str(e)}")
            return False
    
    def find_polygons(self):
        """Абстрактний метод для пошуку багатокутників"""
        raise NotImplementedError("Цей метод має бути перевизначений у підкласах")
    
    def determine_nested_structure(self):
        """Визначає вкладену структуру багатокутників"""
        self.nested_structure = {}
        
        # Спочатку ініціалізуємо структуру
        for i in range(len(self.polygons)):
            self.nested_structure[i] = []
        
        # Для кожної пари багатокутників перевіряємо вкладеність
        for i, outer_polygon in enumerate(self.polygons):
            for j, inner_polygon in enumerate(self.polygons):
                if i != j and outer_polygon.contains_polygon(inner_polygon):
                    # Перевіряємо, чи є прямим батьком
                    is_direct_parent = True
                    for k, middle_polygon in enumerate(self.polygons):
                        if (k != i and k != j and 
                            outer_polygon.contains_polygon(middle_polygon) and 
                            middle_polygon.contains_polygon(inner_polygon)):
                            is_direct_parent = False
                            break
                            
                    if is_direct_parent:
                        self.nested_structure[i].append(j)
        
        return self.nested_structure

class GrahamScanProcessor(PolygonProcessor):
    """Клас для обробки багатокутників за допомогою алгоритму Грема"""
    def __init__(self):
        super().__init__()

    def find_polygons(self):
        """Знаходить багатокутники використовуючи алгоритм Грехем-скан"""
        start_time = time.time()
        
        if not self.all_points:
            return 0
        
        # Копія всіх точок для роботи
        points = self.all_points.copy()
        self.polygons = []
        
        # Спочатку знаходимо зовнішню оболонку всіх точок
        outer_hull_points = self._graham_scan(points)
        if len(outer_hull_points) >= 3:
            outer_polygon = Polygon(outer_hull_points, id=len(self.polygons))
            self.polygons.append(outer_polygon)
            
            # Створюємо множину точок, які використані в зовнішній оболонці
            used_points = set()
            for point in outer_hull_points:
                for i, p in enumerate(points):
                    if p.x == point.x and p.y == point.y:
                        used_points.add(i)
        
        # Продовжуємо шукати внутрішні багатокутники
        remaining_points = [p for i, p in enumerate(points) if i not in used_points]
        
        while len(remaining_points) >= 3:
            # Застосовуємо алгоритм до залишених точок
            hull_points = self._graham_scan(remaining_points)
            
            # Перевіряємо, чи є достатньо точок для формування багатокутника
            if len(hull_points) >= 3:
                polygon = Polygon(hull_points, id=len(self.polygons))
                self.polygons.append(polygon)
                
                # Оновлюємо множину використаних точок
                new_used_points = set()
                for point in hull_points:
                    for i, p in enumerate(remaining_points):
                        if p.x == point.x and p.y == point.y and i not in new_used_points:
                            new_used_points.add(i)
                
                # Оновлюємо список точок, що залишилися
                remaining_points = [p for i, p in enumerate(remaining_points) if i not in new_used_points]
            else:
                break
        
        # Визначаємо вкладену структуру
        self.determine_nested_structure()
        
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def _graham_scan(self, points):
        """Реалізує алгоритм Грехем-скан для знаходження опуклої оболонки"""
        if len(points) < 3:
            return points
        
        # Знаходимо точку з найменшою y-координатою (і найменшою x при рівності)
        lowest_point = min(points, key=lambda p: (p.y, p.x))
        
        # Сортуємо точки за полярним кутом відносно найнижчої точки
        def polar_angle(p):
            return np.arctan2(p.y - lowest_point.y, p.x - lowest_point.x)
        
        sorted_points = sorted(points, key=polar_angle)
        
        # Створюємо стек для алгоритму Грема
        hull = []
        
        # Додаємо перші дві точки
        hull.append(sorted_points[0])
        
        # Поки є принаймні дві точки в стеку і наступна точка не утворює лівий поворот
        for point in sorted_points[1:]:
            while len(hull) > 1 and self._cross_product(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
        
        return hull
    
    def _cross_product(self, p1, p2, p3):
        """Обчислює векторний добуток для визначення повороту"""
        return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    
    def _remove_intersecting_polygons(self):
        """Видаляє багатокутники, які перетинаються з іншими"""
        i = 0
        while i < len(self.polygons):
            j = i + 1
            while j < len(self.polygons):
                if self.polygons[i].intersects_with(self.polygons[j]):
                    if self.polygons[i].area < self.polygons[j].area:
                        self.polygons.pop(i)
                        j = i + 1  
                    else:
                        self.polygons.pop(j)
                else:
                    j += 1
            i += 1


class JarvisMarchProcessor(PolygonProcessor):
    """Клас для обробки багатокутників за допомогою алгоритму Джарвіса (обход подарунка)"""
    def __init__(self):
        super().__init__()
    
    def find_polygons(self):
        """Знаходить багатокутники використовуючи алгоритм Джарвіса"""
        start_time = time.time()
        
        if not self.all_points:
            return 0
        
        points = self.all_points.copy()
        self.polygons = []
        
        # Використовуємо розділення точок на групи для пошуку вкладених багатокутників
        points_used = set()
        
        # Спочатку шукаємо зовнішню оболонку всіх точок
        hull_points = self._jarvis_march(points)
        if len(hull_points) >= 3:
            outer_polygon = Polygon(hull_points, id=len(self.polygons))
            self.polygons.append(outer_polygon)
            
            # Позначаємо використані точки
            for point in hull_points:
                points_used.add(self._find_point_index(points, point))
        
        # Шукаємо внутрішні багатокутники
        remaining_points = [p for i, p in enumerate(points) if i not in points_used]
        
        while remaining_points:
            # Знаходимо опуклу оболонку для залишених точок
            inner_hull = self._jarvis_march(remaining_points)
            
            # Перевіряємо, чи є достатньо точок для формування багатокутника
            if len(inner_hull) >= 3:
                polygon = Polygon(inner_hull, id=len(self.polygons))
                self.polygons.append(polygon)
                
                # Оновлюємо використані точки
                for point in inner_hull:
                    point_idx = self._find_point_index(remaining_points, point)
                    if point_idx is not None:
                        remaining_points.pop(point_idx)
            else:
                break
        
        # Видаляємо багатокутники, які перетинаються з іншими
        self._remove_intersecting_polygons()
        
        # Визначаємо вкладену структуру
        self.determine_nested_structure()
        
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def _jarvis_march(self, points):
        """Реалізує алгоритм Джарвіса для знаходження опуклої оболонки"""
        if len(points) < 3:
            return points
        
        # Знаходимо точку з найменшою x-координатою (і найменшою y при рівності)
        leftmost = min(points, key=lambda p: (p.x, p.y))
        
        hull = []
        current_point = leftmost
        hull.append(current_point)
        
        while True:
            # Початкова наступна точка - будь-яка точка, крім поточної
            next_point = points[0] if points[0] != current_point else points[1]
            
            for point in points:
                # Обчислюємо орієнтацію трьох точок
                orientation = self._orientation(current_point, next_point, point)
                
                # Якщо точка знаходиться справа від поточного шляху або на одній прямій, але далі, то вона стає наступною точкою
                if (next_point == current_point or 
                    orientation > 0 or 
                    (orientation == 0 and 
                     current_point.distance_to(point) > current_point.distance_to(next_point))):
                    next_point = point
            
            # Якщо ми повернулися до початкової точки, завершуємо роботу
            if next_point == leftmost:
                break
                
            hull.append(next_point)
            current_point = next_point
        
        return hull
    
    def _orientation(self, p, q, r):
        """Обчислює орієнтацію трьох точок
        Повертає додатне значення, якщо r знаходиться зліва від прямої pq,
        від'ємне - якщо справа, 0 - якщо на одній прямій"""
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0  
        return 1 if val > 0 else -1  
    
    def _find_point_index(self, points_list, target_point):
        """Знаходить індекс точки у списку"""
        for i, point in enumerate(points_list):
            if point.x == target_point.x and point.y == target_point.y:
                return i
        return None
    
    def _remove_intersecting_polygons(self):
        """Видаляє багатокутники, які перетинаються з іншими"""
        i = 0
        while i < len(self.polygons):
            j = i + 1
            while j < len(self.polygons):
                if self.polygons[i].intersects_with(self.polygons[j]):
                    if self.polygons[i].area < self.polygons[j].area:
                        self.polygons.pop(i)
                        j = i + 1  
                    else:
                        self.polygons.pop(j)
                else:
                    j += 1
            i += 1


class PolygonVisualization:
    """Клас для візуалізації багатокутників"""
    def __init__(self, canvas_frame):
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def clear(self):
        """Очищає область графіка"""
        self.plot.clear()
        self.canvas.draw()
        
    def plot_points(self, points):
        """Відображає точки на графіку"""
        self.plot.clear()
        
        x_coords = [point.x for point in points]
        y_coords = [point.y for point in points]
        
        self.plot.scatter(x_coords, y_coords, c='blue')
        
        # Відображення номерів точок
        for point in points:
            self.plot.text(point.x, point.y, str(point.id), fontsize=9)
        
        self.plot.set_title('Точки')
        self.plot.grid(True)
        self.plot.set_aspect('equal')
        self.canvas.draw()
        
    def draw_polygons(self, polygons, nested_structure=None, elapsed_time=None):
        """Відображає багатокутники з урахуванням вкладеної структури"""
        self.plot.clear()
        
        if not polygons:
            self.plot.set_title('Багатокутники не знайдено')
            self.canvas.draw()
            return
        
        # Генеруємо кольори для багатокутників
        cmap = plt.cm.get_cmap('tab10', len(polygons))
        colors = [cmap(i) for i in range(len(polygons))]
        
        polygons_sorted = sorted(polygons, key=lambda p: p.area, reverse=True)
        
        # Спочатку відображаємо багатокутники
        for i, polygon in enumerate(polygons_sorted):
            x_coords = polygon.get_x_coords()
            y_coords = polygon.get_y_coords()

            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])
            
            # Використовуємо різні кольори та прозорість для кожного багатокутника
            color = colors[i % len(colors)]
            alpha = 0.3  # Прозорість заповнення
            
            self.plot.fill(x_coords, y_coords, alpha=alpha, color=color)
            self.plot.plot(x_coords, y_coords, '-o', color=color, lw=1.5, label=f'Багатокутник {i+1}')
            
            # Додаємо підпис центру багатокутника
            center = polygon.get_center()
            if center:
                self.plot.text(center.x, center.y, f'P{i+1}', fontsize=12, 
                              ha='center', va='center', fontweight='bold')
            
            # Відображення номерів вершин
            for j, (x, y) in enumerate(zip(polygon.get_x_coords(), polygon.get_y_coords())):
                self.plot.text(x, y, f'{polygon.points[j].id}', fontsize=9)
        
        # Додаємо відображення часу виконання алгоритму
        if elapsed_time is not None:
            self.plot.text(0.02, 0.98, f'Час роботи алгоритму: {elapsed_time:.6f} с', 
                         transform=self.plot.transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                                                           facecolor='white', alpha=0.8))
        
        self.plot.set_title('Вкладені опуклі багатокутники')
        self.plot.grid(True)
        self.plot.set_aspect('equal')
        self.plot.legend(loc='upper right')
        
        self.plot.autoscale()
        
        self.canvas.draw()


class PolygonApp:
    """Головний клас програми"""
    def __init__(self, root):
        self.root = root
        self.root.title("Обробка вкладених опуклих багатокутників - Варіант №8")
        self.root.geometry("900x700")
        
        self.graham_processor = GrahamScanProcessor()
        self.jarvis_processor = JarvisMarchProcessor()
        self.current_processor = self.graham_processor
        
        self.create_widgets()
        
    def create_widgets(self):
        """Створює графічний інтерфейс"""
        # Головний фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Верхній фрейм для кнопок
        top_frame = ttk.LabelFrame(main_frame, text="Керування", padding="5")
        top_frame.pack(fill=tk.X, pady=5)
        
        # Кнопки верхнього фрейму
        load_button = ttk.Button(top_frame, text="Завантажити точки", command=self.load_points)
        load_button.grid(row=0, column=0, padx=5, pady=5)
        
        generate_button = ttk.Button(top_frame, text="Згенерувати випадкові точки", command=self.generate_random_points)
        generate_button.grid(row=0, column=1, padx=5, pady=5)
        
        save_button = ttk.Button(top_frame, text="Зберегти точки у файл", command=self.save_points)
        save_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Випадаючий список алгоритмів
        algorithm_label = ttk.Label(top_frame, text="Алгоритм:")
        algorithm_label.grid(row=0, column=3, padx=5, pady=5)
        
        self.algorithm_var = tk.StringVar(value="Graham Scan")
        algorithm_dropdown = ttk.Combobox(top_frame, textvariable=self.algorithm_var, 
                                         values=["Graham Scan", "Jarvis March"])
        algorithm_dropdown.grid(row=0, column=4, padx=5, pady=5)
        algorithm_dropdown.bind("<<ComboboxSelected>>", self.change_algorithm)
        
        # Кнопка для запуску алгоритму
        process_button = ttk.Button(top_frame, text="Обробити", command=self.process_points)
        process_button.grid(row=0, column=5, padx=5, pady=5)
        
        # Кнопка для порівняння алгоритмів
        compare_button = ttk.Button(top_frame, text="Порівняти алгоритми", command=self.compare_algorithms)
        compare_button.grid(row=0, column=6, padx=5, pady=5)
        
        # Фрейм для візуалізації
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Створюємо об'єкт для візуалізації
        self.visualization = PolygonVisualization(canvas_frame)
        
        # Фрейм для інформації
        info_frame = ttk.LabelFrame(main_frame, text="Інформація", padding="5")
        info_frame.pack(fill=tk.X, pady=5)
        
        # Інформаційний текст
        self.info_text = tk.Text(info_frame, height=5, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, expand=True)
        self.info_text.config(state=tk.DISABLED)
        
        # Статусний рядок
        self.status_var = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.set_status("Готово до роботи")
    
    def set_status(self, message):
        """Встановлює текст статусного рядка"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def update_info(self, message):
        """Оновлює інформаційний текст"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.config(state=tk.DISABLED)
    
    def change_algorithm(self, event=None):
        """Змінює поточний алгоритм обробки"""
        algorithm = self.algorithm_var.get()
        if algorithm == "Graham Scan":
            self.current_processor = self.graham_processor
        else:
            self.current_processor = self.jarvis_processor
        
        self.set_status(f"Вибрано алгоритм: {algorithm}")
    
    def load_points(self):
        """Завантажує точки з файлу"""
        file_path = filedialog.askopenfilename(
            title="Виберіть файл з точками",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        if self.current_processor.load_points_from_file(file_path):
            self.visualization.plot_points(self.current_processor.all_points)
            self.set_status(f"Завантажено {len(self.current_processor.all_points)} точок з файлу")
            self.update_info(f"Завантажено {len(self.current_processor.all_points)} точок з файлу.\n"
                           f"Використовуйте кнопку 'Обробити' для пошуку багатокутників.")
    
    def generate_random_points(self):
        """Генерує випадкові точки"""
        num_points = simpledialog.askinteger("Генерація точок", 
                                           "Введіть кількість точок:", 
                                           minvalue=5, maxvalue=100)
        if not num_points:
            return
        
        # Очищаємо попередні точки
        self.current_processor.all_points = []
        
        # Генеруємо точки
        self._generate_nested_polygon_points(num_points)
        
        # Відображаємо згенеровані точки
        self.visualization.plot_points(self.current_processor.all_points)
        self.set_status(f"Згенеровано {len(self.current_processor.all_points)} випадкових точок")
        self.update_info(f"Згенеровано {len(self.current_processor.all_points)} випадкових точок.\n"
                       f"Використовуйте кнопку 'Обробити' для пошуку багатокутників.")
    
    def _generate_nested_polygon_points(self, total_points):
        """Генерує точки для вкладених багатокутників"""
        num_polygons = random.randint(2, min(4, total_points // 3))
        
        points_per_polygon = self._distribute_points(total_points, num_polygons)
        
        # Розмір області для генерації
        max_size = 100
        center_x, center_y = 0, 0
        
        # Генеруємо точки для кожного багатокутника
        point_id = 1
        for i, num_points in enumerate(points_per_polygon):
            scale = max_size * (num_polygons - i) / num_polygons
            
            if i > 0:
                center_offset = scale * 0.1
                center_x += random.uniform(-center_offset, center_offset)
                center_y += random.uniform(-center_offset, center_offset)

            for _ in range(num_points):
                angle = random.uniform(0, 2 * np.pi)
 
                radius = scale * (0.8 + random.uniform(0, 0.2))
                
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)

                x += random.uniform(-scale*0.05, scale*0.05)
                y += random.uniform(-scale*0.05, scale*0.05)
                
                self.current_processor.all_points.append(Point(x, y, point_id))
                point_id += 1
    
    def _distribute_points(self, total, num_groups):
        """Розподіляє точки між групами"""
        # Мінімальна кількість точок на групу
        min_points = 3

        points_left = total - min_points * num_groups
        base_points = [min_points] * num_groups
        
        for i in range(num_groups):
            weight = (num_groups - i) / sum(range(1, num_groups + 1))
            extra = int(points_left * weight)
            base_points[i] += extra
            points_left -= extra

        i = 0
        while points_left > 0:
            base_points[i % num_groups] += 1
            points_left -= 1
            i += 1
        
        return base_points
    
    def save_points(self):
        """Зберігає точки у файл"""
        if not self.current_processor.all_points:
            messagebox.showwarning("Попередження", "Немає точок для збереження")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Зберегти точки",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for point in self.current_processor.all_points:
                    file.write(f"Point {point.id}: x={point.x}, y={point.y}\n")
                    
            self.set_status(f"Точки збережено у файл: {file_path}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при збереженні файлу: {str(e)}")
    
    def process_points(self):
        """Обробляє точки за допомогою поточного алгоритму"""
        if not self.current_processor.all_points:
            messagebox.showwarning("Попередження", "Спочатку завантажте або згенеруйте точки")
            return
            
        # Інформуємо користувача
        self.set_status("Обробка точок...")
        
        # Запускаємо обробку та вимірюємо час
        elapsed_time = self.current_processor.find_polygons()
        
        # Відображаємо результат
        self.visualization.draw_polygons(self.current_processor.polygons, self.current_processor.nested_structure)
        
        # Оновлюємо інформацію
        algorithm_name = self.algorithm_var.get()
        info_message = (f"Алгоритм: {algorithm_name}\n"
                       f"Кількість багатокутників: {len(self.current_processor.polygons)}\n"
                       f"Час обробки: {elapsed_time:.6f} секунд")
        self.update_info(info_message)
        
        self.set_status(f"Знайдено {len(self.current_processor.polygons)} багатокутників за {elapsed_time:.6f} секунд")
    
    def compare_algorithms(self):
        """Порівнює час виконання обох алгоритмів"""
        if not self.graham_processor.all_points:
            messagebox.showwarning("Попередження", "Спочатку завантажте або згенеруйте точки")
            return
            
        # Запам'ятовуємо початковий вибір алгоритму
        original_algorithm = self.algorithm_var.get()
            
        # Копіюємо точки в обидва процесори
        self.jarvis_processor.all_points = self.graham_processor.all_points.copy()
        
        # Інформуємо користувача
        self.set_status("Порівняння алгоритмів...")
        
        # Вимірюємо час для алгоритму Грема
        graham_time = self.graham_processor.find_polygons()
        graham_polygons = len(self.graham_processor.polygons)
        
        # Вимірюємо час для алгоритму Джарвіса
        jarvis_time = self.jarvis_processor.find_polygons()
        jarvis_polygons = len(self.jarvis_processor.polygons)
        
        # Відновлюємо початковий вибір алгоритму
        if original_algorithm == "Graham Scan":
            self.current_processor = self.graham_processor
            self.visualization.draw_polygons(self.graham_processor.polygons, self.graham_processor.nested_structure)
        else:
            self.current_processor = self.jarvis_processor
            self.visualization.draw_polygons(self.jarvis_processor.polygons, self.jarvis_processor.nested_structure)
        
        # Оновлюємо інформацію з порівнянням
        info_message = (f"Порівняння алгоритмів:\n"
                    f"Graham Scan: {graham_time:.6f} с, {graham_polygons} багатокутників\n"
                    f"Jarvis March: {jarvis_time:.6f} с, {jarvis_polygons} багатокутників\n"
                    f"Різниця: {abs(graham_time - jarvis_time):.6f} с")
        self.update_info(info_message)
        
        # Оновлюємо статус
        faster = "Graham Scan" if graham_time < jarvis_time else "Jarvis March"
        self.set_status(f"Порівняння завершено. Швидший алгоритм: {faster}")
        
        # Відновлюємо вибір у випадаючому списку
        self.algorithm_var.set(original_algorithm)


def main():
    """Головна функція програми"""
    root = tk.Tk()
    app = PolygonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
