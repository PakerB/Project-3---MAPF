import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle, Circle
import numpy as np
import sys
import tkinter as tk

# Nhập thư viện LaCAM2
sys.path.insert(0, 'e:/Prj3/lacam2_py')
from lacam2.graph import Graph, Vertex
from lacam2.instance import Instance
from lacam2.dist_table import DistTable
from lacam2 import solve

# ==================== CẤU HÌNH ====================
MAP_SIZE = 50
NUM_AGENTS = 30  # Giảm từ 50 xuống 30 để giảm lag
OBSTACLE_DENSITY = 0.25
ANIMATION_INTERVAL = 50  # Tăng lên một chút (50ms) để mượt hơn

class MapGenerator:
    """Tạo map với rooms + corridors"""
    
    def __init__(self, size, obstacle_density):
        self.size = size
        self.obstacle_density = obstacle_density
        self.obstacles = set()
        self.free_cells = []
        
    def generate(self):
        """Tạo map rooms+corridors"""
        self.obstacles = set()
        self.free_cells = []
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Tạo rooms
        rooms = []
        room_count = max(4, self.size // 8)
        for _ in range(room_count):
            rw = random.randint(3, min(12, max(3, self.size // 6)))
            rh = random.randint(3, min(12, max(3, self.size // 6)))
            rx = random.randint(1, self.size - rw - 2)
            ry = random.randint(1, self.size - rh - 2)
            rooms.append((rx, ry, rw, rh))
            for r in range(ry, ry + rh):
                for c in range(rx, rx + rw):
                    grid[r][c] = '.'

        # Nối rooms bằng corridors
        centers = [(rx + rw // 2, ry + rh // 2) for rx, ry, rw, rh in rooms]
        random.shuffle(centers)
        for i in range(1, len(centers)):
            x1, y1 = centers[i - 1]
            x2, y2 = centers[i]
            if random.random() < 0.5:
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    grid[y1][x] = '.'
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    grid[y][x2] = '.'
            else:
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    grid[y][x1] = '.'
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    grid[y2][x] = '.'

        # Protected zones
        protected = set()
        for rx, ry, rw, rh in rooms:
            for r in range(ry, ry + rh):
                for c in range(rx, rx + rw):
                    protected.add((r, c))
        for i in range(1, len(centers)):
            x1, y1 = centers[i - 1]
            x2, y2 = centers[i]
            for x in range(min(x1, x2), max(x1, x2) + 1):
                protected.add((y1, x))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                protected.add((y, x2))

        # Thêm obstacles theo clusters
        target_obstacles = int(self.size * self.size * self.obstacle_density)
        cluster_centers = []
        min_sep = max(3, self.size // 12)
        max_centers = max(10, target_obstacles // 6)
        attempts = 0
        while len(cluster_centers) < max_centers and attempts < self.size * self.size * 4:
            r = random.randint(1, self.size - 2)
            c = random.randint(1, self.size - 2)
            if (r, c) in protected:
                attempts += 1
                continue
            ok = True
            for (cr, cc) in cluster_centers:
                if abs(cr - r) + abs(cc - c) < min_sep:
                    ok = False
                    break
            if ok:
                cluster_centers.append((r, c))
            attempts += 1

        # Thêm blobs
        for (cr, cc) in cluster_centers:
            blob_size = random.randint(3, max(3, self.size // 12))
            for _ in range(blob_size * blob_size):
                dr = random.randint(-blob_size//2, blob_size//2)
                dc = random.randint(-blob_size//2, blob_size//2)
                rr, rc = cr + dr, cc + dc
                if 0 <= rr < self.size and 0 <= rc < self.size and (rr, rc) not in protected:
                    grid[rr][rc] = '@'

        # Thêm scattered obstacles
        current_obstacles = sum(1 for r in range(self.size) for c in range(self.size) if grid[r][c] == '@')
        to_add = max(0, target_obstacles - current_obstacles)
        attempts = 0
        while to_add > 0 and attempts < self.size * self.size * 4:
            r = random.randint(0, self.size - 1)
            c = random.randint(0, self.size - 1)
            if grid[r][c] == '.' and (r, c) not in protected:
                grid[r][c] = '@'
                to_add -= 1
            attempts += 1

        # Finalize
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] == '@':
                    self.obstacles.add((r, c))
                else:
                    self.free_cells.append((r, c))

        return self.obstacles, self.free_cells

    def get_largest_component(self):
        """Lấy connected component lớn nhất"""
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        free = set((r,c) for r in range(self.size) for c in range(self.size) if (r,c) not in self.obstacles)
        visited = set()
        best_comp = set()
        for cell in free:
            if cell in visited:
                continue
            comp = set()
            stack = [cell]
            visited.add(cell)
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr+dr, cc+dc
                    if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) in free and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
            if len(comp) > len(best_comp):
                best_comp = comp
        return best_comp
    
    def to_lacam2_format(self):
        """Chuyển sang định dạng LaCAM2 map file"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        for r, c in self.obstacles:
            grid[r][c] = '@'
        lines = [f"type octile", f"height {self.size}", f"width {self.size}", "map"]
        for row in grid:
            lines.append(''.join(row))
        return '\n'.join(lines)

class LaCAM2Visualizer:
    """Visualizer LaCAM2 - UI như bản gốc"""
    
    def __init__(self, map_size=MAP_SIZE, num_agents=NUM_AGENTS, obstacle_density=OBSTACLE_DENSITY):
        self.map_size = map_size
        self.num_agents = num_agents
        self.obstacle_density = obstacle_density
        
        # Tạo map và solve
        print(f"Đang tạo map {map_size}x{map_size} với {num_agents} agents...")
        self.map_gen = MapGenerator(map_size, obstacle_density)

        max_attempts = 6
        self.solution = None
        for attempt in range(1, max_attempts + 1):
            print(f"-- Attempt {attempt}/{max_attempts}...")
            self.obstacles, self.free_cells = self.map_gen.generate()
            largest = self.map_gen.get_largest_component()
            print(f"Map: {len(self.free_cells)} free, {len(self.obstacles)} obstacles, largest_component={len(largest)}")

            if len(largest) < num_agents * 2:
                print(f"  -> Largest component quá nhỏ, tạo lại")
                continue

            # Chọn starts/goals từ largest component
            pts = random.sample(list(largest), num_agents * 2)
            self.starts = pts[:num_agents]
            self.goals = pts[num_agents:]

            print("Đang giải bằng LaCAM2...")
            self.solution = self.solve_lacam2()
            if self.solution:
                print(f"[OK] Solution: {len(self.solution)} steps")
                swap_count = self.count_swaps()
                if swap_count > 0:
                    print(f"[WARNING] {swap_count} swaps detected!")
                else:
                    print("[OK] No swaps - correct!")
                break
            else:
                print("  -> No solution, retry\n")

        if not self.solution:
            print("[ERROR] No solution found")
            sys.exit(1)

        # Auto-detect màn hình và tính toán kích thước
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Tính DPI và figsize để fit màn hình
        dpi = 100
        fig_width = screen_width / dpi * 0.95  # 95% màn hình
        fig_height = screen_height / dpi * 0.9
        
        print(f"Screen: {screen_width}x{screen_height}, Figure: {fig_width:.1f}x{fig_height:.1f} inches")
        
        # Setup animation - NỀN TRẮNG với buttons bên phải
        self.fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white', dpi=dpi)
        
        # Map chiếm 70% bên trái, buttons 30% bên phải
        self.ax = self.fig.add_axes([0.05, 0.1, 0.62, 0.85])
        self.ax.set_facecolor('white')
        self.fig.canvas.manager.set_window_title('LaCAM2 Visualizer - No Swap')
        
        # Maximize window
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except:
            try:
                mng.frame.Maximize(True)
            except:
                pass
        
        self.colors = plt.cm.tab20(np.linspace(0, 1, num_agents))
        self.current_step = 0
        self.paused = True
        self.finished = False
        self.anim = None  # Lưu animation object
        
        # Các nút bên phải
        self.setup_ui()
        
        # Vẽ frame đầu tiên
        self.draw_frame(0)
        
    def solve_lacam2(self):
        """Giải bằng LaCAM2"""
        try:
            map_content = self.map_gen.to_lacam2_format()
            map_file = 'e:/Prj3/lacam2_py/temp_map.map'
            with open(map_file, 'w') as f:
                f.write(map_content)
            
            scen_lines = ["version 1"]
            for i in range(self.num_agents):
                sr, sc = self.starts[i]
                gr, gc = self.goals[i]
                scen_lines.append(f"0\ttemp_map.map\t{self.map_size}\t{self.map_size}\t{sc}\t{sr}\t{gc}\t{gr}\t0")
            
            scen_file = 'e:/Prj3/lacam2_py/temp_scen.scen'
            with open(scen_file, 'w') as f:
                f.write('\n'.join(scen_lines))
            
            ins = Instance(map_file, scen_file, self.num_agents)
            from lacam2.utils import Deadline
            import time
            deadline = Deadline(time.time() * 1000 + 30000)
            solution = solve(ins, verbose=0, deadline=deadline)
            
            if not solution:
                return None
            
            # Chuyển đổi solution
            path = []
            for config in solution:
                step = []
                for v in config:
                    idx = v.index
                    row = idx // self.map_size
                    col = idx % self.map_size
                    step.append((row, col))
                path.append(step)
            
            return path
            
        except Exception as e:
            print(f"Error solving: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def count_swaps(self):
        """Đếm swaps trong solution"""
        if not self.solution or len(self.solution) < 2:
            return 0
        
        swap_count = 0
        for t in range(len(self.solution) - 1):
            config_t = self.solution[t]
            config_t1 = self.solution[t + 1]
            
            for i in range(len(config_t)):
                for j in range(i + 1, len(config_t)):
                    if (config_t[i] == config_t1[j] and config_t[j] == config_t1[i]):
                        swap_count += 1
                        print(f"  Swap at step {t}: agent {i} <-> agent {j}")
        
        return swap_count
    
    def setup_ui(self):
        """Tạo các nút bên phải - VỊ TRÍ CỐ ĐỊNH"""
        # Các nút ở bên phải (70% width = vùng map, các nút từ 72% trở đi)
        button_width = 0.15
        button_height = 0.05
        button_x = 0.75  # Bắt đầu từ 75% (sau map 70%)
        
        # Nút Play/Pause (ở trên)
        ax_pause = self.fig.add_axes([button_x, 0.75, button_width, button_height])
        self.btn_pause = Button(ax_pause, 'Play', color='lightgreen', hovercolor='green')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        # Nút Reset
        ax_reset = self.fig.add_axes([button_x, 0.65, button_width, button_height])
        self.btn_reset = Button(ax_reset, 'Reset', color='lightgray', hovercolor='gray')
        self.btn_reset.on_clicked(self.reset_animation)
        
        # Nút New Map
        ax_newmap = self.fig.add_axes([button_x, 0.55, button_width, button_height])
        self.btn_newmap = Button(ax_newmap, 'New Map', color='lightblue', hovercolor='blue')
        self.btn_newmap.on_clicked(self.new_map)
        
        # Hộp thông tin
        info_text = f"Map: {self.map_size}x{self.map_size}\nAgents: {self.num_agents}\nObstacles: {len(self.obstacles)}\nSolution: {len(self.solution)} steps"
        self.fig.text(button_x + 0.02, 0.35, info_text, fontsize=12, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    
    def toggle_pause(self, event):
        self.paused = not self.paused
        self.btn_pause.label.set_text('Pause' if not self.paused else 'Play')
        if self.finished and not self.paused:
            self.current_step = 0
            self.finished = False
    
    def reset_animation(self, event):
        self.current_step = 0
        self.paused = True
        self.finished = False
        self.btn_pause.label.set_text('Play')
        self.draw_frame(0)
    
    def new_map(self, event):
        """Tạo map mới - restart toàn bộ"""
        print("\n=== Tạo map mới ===")
        plt.close('all')  # Đóng tất cả figures
        
        # Tạo visualizer mới và chạy trong cùng process
        import sys
        sys.stdout.flush()
        
        # Re-run main
        main()
    
    def draw_frame(self, frame_num):
        """Vẽ frame - TỐI ƯU: chỉ vẽ lại agents và trails"""
        # Chỉ clear khi cần thiết
        if frame_num == 0 or self.current_step == 0:
            self.ax.clear()
            self.ax.set_facecolor('white')
            
            # Grid (xám nhạt) - chỉ vẽ 1 lần
            for i in range(self.map_size + 1):
                self.ax.plot([0, self.map_size], [i, i], 'lightgray', linewidth=0.5)
                self.ax.plot([i, i], [0, self.map_size], 'lightgray', linewidth=0.5)
            
            # Obstacles (đen) - chỉ vẽ 1 lần
            for r, c in self.obstacles:
                self.ax.add_patch(Rectangle((c, r), 1, 1, color='black'))
            
            # Goals (vuông nhỏ màu nhạt) - chỉ vẽ 1 lần
            for i, (gr, gc) in enumerate(self.goals):
                self.ax.add_patch(Rectangle((gc + 0.3, gr + 0.3), 0.4, 0.4, 
                                               color=self.colors[i], alpha=0.3))
            
            self.ax.set_xlim(0, self.map_size)
            self.ax.set_ylim(0, self.map_size)
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            # Chỉ xóa agents và trails cũ
            for artist in self.ax.patches[len(self.obstacles) + len(self.goals):]:
                artist.remove()
            for line in self.ax.lines[len(self.obstacles) * 2 + self.map_size * 2 + 2:]:
                line.remove()
            for text in self.ax.texts:
                text.remove()
        
        # Vẽ agents
        current_positions = self.solution[self.current_step]
        for i, (r, c) in enumerate(current_positions):
            self.ax.add_patch(Circle((c + 0.5, r + 0.5), 0.35, 
                                        color=self.colors[i], zorder=10))
            # Chỉ hiện ID khi có ít agents (giảm lag)
            if self.num_agents <= 30:
                self.ax.text(c + 0.5, r + 0.5, str(i), 
                            ha='center', va='center', fontsize=8, color='white', weight='bold')
        
        # Trail (chỉ 5 steps để giảm lag)
        if self.current_step > 0:
            for i in range(self.num_agents):
                trail_x = []
                trail_y = []
                for t in range(max(0, self.current_step - 5), self.current_step + 1):
                    r, c = self.solution[t][i]
                    trail_x.append(c + 0.5)
                    trail_y.append(r + 0.5)
                self.ax.plot(trail_x, trail_y, color=self.colors[i], 
                           alpha=0.3, linewidth=1, linestyle='--')
        
        # Title
        status = "FINISHED" if self.finished else f"Step {self.current_step}/{len(self.solution)-1}"
        self.ax.set_title(f'LaCAM2 (No Swap) - {status}\n'
                         f'Map: {self.map_size}x{self.map_size}, Obstacles: {len(self.obstacles)}, Agents: {self.num_agents}', 
                         fontsize=14, weight='bold', color='black')
        
        # DỪNG KHI ĐẾN CUỐI
        if not self.paused and not self.finished:
            self.current_step += 1
            if self.current_step >= len(self.solution):
                self.current_step = len(self.solution) - 1
                self.finished = True
                self.paused = True
                self.btn_pause.label.set_text('Play')
                print("[OK] Finished! Click Play to replay or New Map.")
    
    def run(self):
        """Run animation"""
        self.anim = animation.FuncAnimation(
            self.fig, 
            self.draw_frame,
            interval=ANIMATION_INTERVAL,
            repeat=True,
            cache_frame_data=False,
            blit=False  # Tắt blit để tương thích tốt hơn
        )
        print("\n>>> Window opened! Click Play to start.")
        plt.show()
        return self.anim

def main():
    print("="*60)
    print("LaCAM2 VISUALIZER - FULL ALGORITHM (NO SWAP)")
    print("="*60)
    print(f"Map: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Agents: {NUM_AGENTS}")
    print(f"Obstacle density: {OBSTACLE_DENSITY}")
    print("="*60)
    
    try:
        visualizer = LaCAM2Visualizer(MAP_SIZE, NUM_AGENTS, OBSTACLE_DENSITY)
        visualizer.run()
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
