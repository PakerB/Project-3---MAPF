"""
Định nghĩa đồ thị cho LaCAM2
"""

import re
from typing import List, Optional


class Vertex:
    """Đỉnh của đồ thị đại diện cho một vị trí"""
    
    def __init__(self, vertex_id: int, index: int):
        self.id = vertex_id  # chỉ số cho V trong Graph
        self.index = index   # chỉ số cho U, width * y + x
        self.neighbor: List['Vertex'] = []
    
    def __repr__(self):
        return str(self.index)
    
    def __str__(self):
        return str(self.index)


Vertices = List[Optional[Vertex]]
Config = List[Vertex]  # vị trí của tất cả agent


class Graph:
    """Biểu diễn đồ thị dạng lưới"""
    
    def __init__(self, filename: Optional[str] = None):
        self.V: List[Vertex] = [] 
        self.U: List[Optional[Vertex]] = []  
        self.width: int = 0
        self.height: int = 0
        
        if filename:
            self._load_from_file(filename)
    
    def _load_from_file(self, filename: str):
        """Load đồ thị từ file map"""
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"File {filename} is not found.")
            return
        
        r_height = re.compile(r'height\s(\d+)')
        r_width = re.compile(r'width\s(\d+)')
        r_map = re.compile(r'map')
        
        line_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            
            match = r_height.match(line)
            if match:
                self.height = int(match.group(1))
            
            match = r_width.match(line)
            if match:
                self.width = int(match.group(1))
            
            if r_map.match(line):
                line_idx = i + 1
                break
        
        self.U = [None] * (self.width * self.height)
        
        # Tạo các đỉnh
        y = 0
        for i in range(line_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
                
            for x in range(min(len(line), self.width)):
                s = line[x]
                if s == 'T' or s == '@':  # chướng ngại vật
                    continue
                
                index = self.width * y + x
                v = Vertex(len(self.V), index)
                self.V.append(v)
                self.U[index] = v
            
            y += 1
            if y >= self.height:
                break
        
        # Tạo các cạnh 
        for y in range(self.height):
            for x in range(self.width):
                v = self.U[self.width * y + x]
                if v is None:
                    continue
                
                # Left
                if x > 0:
                    u = self.U[self.width * y + (x - 1)]
                    if u is not None:
                        v.neighbor.append(u)
                
                # right
                if x < self.width - 1:
                    u = self.U[self.width * y + (x + 1)]
                    if u is not None:
                        v.neighbor.append(u)
                
                # top
                if y < self.height - 1:
                    u = self.U[self.width * (y + 1) + x]
                    if u is not None:
                        v.neighbor.append(u)
                
                # bot
                if y > 0:
                    u = self.U[self.width * (y - 1) + x]
                    if u is not None:
                        v.neighbor.append(u)
    
    def size(self) -> int:
        """Lấy số lượng đỉnh"""
        return len(self.V)


def is_same_config(c1: Config, c2: Config) -> bool:
    """Kiểm tra xem hai cấu hình có giống nhau không"""
    if len(c1) != len(c2):
        return False
    return all(v1.id == v2.id for v1, v2 in zip(c1, c2))


class ConfigHasher:
    """Hàm băm cho cấu hình"""
    
    @staticmethod
    def hash(config: Config) -> int:
        """Tính giá trị băm của cấu hình"""
        h = len(config)
        for v in config:
            h ^= v.id + 0x9e3779b9 + (h << 6) + (h >> 2)
        return h


def config_to_str(config: Config) -> str:
    """Chuyển cấu hình thành chuỗi"""
    return f"<{','.join(f'{v.index:5d}' for v in config)}>"
