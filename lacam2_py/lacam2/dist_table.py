"""
Bảng khoảng cách với tính toán lazy sử dụng BFS
"""

from typing import List, Optional
from collections import deque
from .graph import Vertex
from .instance import Instance


class DistTable:
    """Bảng khoảng cách tính toán bằng BFS với lazy evaluation"""
    
    def __init__(self, ins: Instance):
        """
        Khởi tạo bảng khoảng cách
        
        Tham số:
            ins: Instance của bài toán
        """
        self.V_size = ins.G.size()
        # dist = [agent_id][vertex_id] 
        self.table: List[List[Optional[int]]] = [[None] * self.V_size for _ in range(ins.N)]
        self.OPEN: List[deque] = [deque() for _ in range(ins.N)]
        
        self.setup(ins)
    
    def setup(self, ins: Instance):
        """Khởi tạo bảng khoảng cách với vị trí đích"""
        for i in range(ins.N):
            g = ins.goals[i]
            self.table[i][g.id] = 0
            self.OPEN[i].append(g)
    
    def get(self, agent_id: int, v: Vertex) -> int:
        """
        Lấy khoảng cách từ đích của agent đến đỉnh v
        Sử dụng lazy evaluation với BFS
        
        Tham số:
            agent_id: Chỉ số agent
            v: Đỉnh đích (có thể là Vertex hoặc vertex_id)
        
        Trả về:
            Giá trị khoảng cách
        """
        if isinstance(v, Vertex):
            v_id = v.id
            vertex = v
        else:
            v_id = v
            vertex = None
        
        if self.table[agent_id][v_id] is not None:
            return self.table[agent_id][v_id]  
        
        # Tính toán bằng BFS
        while self.OPEN[agent_id]:
            u = self.OPEN[agent_id].popleft()
            dist_u = self.table[agent_id][u.id]
            
            for neighbor in u.neighbor:
                if self.table[agent_id][neighbor.id] is None:
                    self.table[agent_id][neighbor.id] = dist_u + 1
                    self.OPEN[agent_id].append(neighbor)
            
            # Kiểm tra xem đã tìm thấy đích chưa
            if self.table[agent_id][v_id] is not None:
                return self.table[agent_id][v_id]  
        
        return self.V_size * 2
