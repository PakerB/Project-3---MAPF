"""
Định nghĩa instance (bài toán) cho LaCAM2
"""

import re
import random
from typing import List, Optional
from .graph import Graph, Config, Vertex
from .utils import info


class Instance:
    """Instance của bài toán MAPF"""
    
    def __init__(self, map_filename: str, 
                 scen_filename: Optional[str] = None,
                 num_agents: int = 1,
                 start_indexes: Optional[List[int]] = None,
                 goal_indexes: Optional[List[int]] = None,
                 mt: Optional[random.Random] = None):
        """
        Khởi tạo instance
        
        Tham số:
            map_filename: Đường dẫn đến file map
            scen_filename: Đường dẫn đến file scenario (tùy chọn)
            num_agents: Số lượng agent
            start_indexes: Vị trí bắt đầu (dùng cho test)
            goal_indexes: Vị trí đích (dùng cho test)
            mt: Bộ sinh số ngẫu nhiên (dùng cho instance ngẫu nhiên)
        """
        self.G = Graph(map_filename)
        self.starts: Config = []
        self.goals: Config = []
        self.N = num_agents
        
        if start_indexes is not None and goal_indexes is not None:
            self._init_from_indexes(start_indexes, goal_indexes)
        elif scen_filename:
            self._load_scenario(scen_filename)
        elif mt is not None:
            self._generate_random(mt)
        else:
            raise ValueError("Must provide either scenario file or random generator")
    
    def _init_from_indexes(self, start_indexes: List[int], goal_indexes: List[int]):
        """Khởi tạo từ danh sách chỉ số"""
        for k in start_indexes:
            v = self.G.U[k]
            if v is not None:
                self.starts.append(v)
        for k in goal_indexes:
            v = self.G.U[k]
            if v is not None:
                self.goals.append(v)
        self.N = len(start_indexes)
    
    def _load_scenario(self, scen_filename: str):
        """Load các cặp start-goal từ file scenario"""
        try:
            with open(scen_filename, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            info(0, 0, f"{scen_filename} is not found")
            return
        
        r_instance = re.compile(r'\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+')
        
        for line in lines:
            line = line.strip()
            match = r_instance.match(line)
            
            if match:
                x_s = int(match.group(1))
                y_s = int(match.group(2))
                x_g = int(match.group(3))
                y_g = int(match.group(4))
                
                # Kiểm tra giới hạn
                if x_s < 0 or x_s >= self.G.width or x_g < 0 or x_g >= self.G.width:
                    break
                if y_s < 0 or y_s >= self.G.height or y_g < 0 or y_g >= self.G.height:
                    break
                
                s = self.G.U[self.G.width * y_s + x_s]
                g = self.G.U[self.G.width * y_g + x_g]
                
                if s is None or g is None:
                    break
                
                self.starts.append(s)
                self.goals.append(g)
            
            if len(self.starts) >= self.N:
                break
    
    def _generate_random(self, mt: random.Random):
        """Tạo các cặp start-goal ngẫu nhiên"""
        v_size = self.G.size()
        
        # Đặt vị trí bắt đầu
        s_indexes = list(range(v_size))
        mt.shuffle(s_indexes)
        
        for i in range(min(self.N, v_size)):
            self.starts.append(self.G.V[s_indexes[i]])
        
        # Đặt vị trí đích
        g_indexes = list(range(v_size))
        mt.shuffle(g_indexes)
        
        for i in range(min(self.N, v_size)):
            self.goals.append(self.G.V[g_indexes[i]])
    
    def is_valid(self, verbose: int = 0) -> bool:
        """Kiểm tra tính hợp lệ đơn giản của instance"""
        if self.N != len(self.starts) or self.N != len(self.goals):
            info(1, verbose, "invalid N, check instance")
            return False
        return True


Solution = List[Config]


def solution_to_str(solution: Solution) -> str:
    """Chuyển solution thành chuỗi"""
    if not solution:
        return ""
    
    N = len(solution[0])
    result = []
    
    for i in range(N):
        path = f"{i:5d}:"
        for t, config in enumerate(solution):
            if t > 0:
                path += "->"
            path += f"{config[i].index:5d}"
        result.append(path)
    
    return "\n".join(result)
