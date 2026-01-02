"""
LaCAM* planner implementation
"""

import random
from typing import List, Optional, Set
from collections import deque
from enum import IntEnum

from .instance import Instance, Solution
from .graph import Config, Vertex, is_same_config, ConfigHasher
from .dist_table import DistTable
from .utils import Deadline, is_expired, get_random_float, get_random_int


class Objective(IntEnum):
    """Objective function types"""
    OBJ_NONE = 0
    OBJ_MAKESPAN = 1
    OBJ_SUM_OF_LOSS = 2


class Agent:
    """PIBT agent"""
    
    def __init__(self, agent_id: int):
        self.id = agent_id
        self.v_now: Optional[Vertex] = None
        self.v_next: Optional[Vertex] = None


class LNode:
    """Low-level search node"""
    
    def __init__(self, parent: Optional['LNode'] = None, i: int = 0, v: Optional[Vertex] = None):
        if parent is None:
            self.who: List[int] = []
            self.where: List[Vertex] = []
        else:
            self.who = parent.who.copy()
            self.where = parent.where.copy()
            if v is not None:
                self.who.append(i)
                self.where.append(v)
        
        self.depth = len(self.who)


class HNode:
    """High-level search node"""
    
    HNODE_CNT = 0
    
    def __init__(self, config: Config, D: DistTable, parent: Optional['HNode'], 
                 g: int, h: int):
        HNode.HNODE_CNT += 1
        self.C = config
        self.parent = parent
        self.neighbor: Set['HNode'] = set()
        self.g = g
        self.h = h
        self.f = g + h
        
        # Cho tìm kiếm low-level
        N = len(config)
        self.priorities: List[float] = [0.0] * N
        self.order: List[int] = list(range(N))
        self.search_tree: deque = deque()
        
        # Thiết lập độ ưu tiên
        if parent is None:
            for i in range(N):
                self.priorities[i] = float(D.get(i, config[i])) / N
        else:
            for i in range(N):
                if D.get(i, config[i]) != 0:
                    self.priorities[i] = parent.priorities[i] + 1
                else:
                    self.priorities[i] = parent.priorities[i] - int(parent.priorities[i])
        
        self.order.sort(key=lambda i: -self.priorities[i])  # thứ tự giảm dần
        
        self.search_tree.append(LNode())
        
        if parent is not None:
            parent.neighbor.add(self)


class Planner:
    """LaCAM* planner"""
    
    def __init__(self, ins: Instance, deadline: Optional[Deadline] = None,
                 mt: Optional[random.Random] = None, verbose: int = 0,
                 objective: Objective = Objective.OBJ_NONE,
                 restart_rate: float = 0.001):
        self.ins = ins
        self.deadline = deadline
        self.MT = mt if mt is not None else random.Random()
        self.verbose = verbose
        self.objective = objective
        self.RESTART_RATE = restart_rate
        
        self.N = ins.N
        self.V_size = ins.G.size()
        self.D = DistTable(ins)
        self.loop_cnt = 0
        
        self.C_next: List[List[Optional[Vertex]]] = [[None] * 5 for _ in range(self.N)]
        self.tie_breakers: List[float] = [0.0] * self.V_size  
        self.A: List[Agent] = [Agent(i) for i in range(self.N)]
        self.occupied_now: List[Optional[Agent]] = [None] * self.V_size
        self.occupied_next: List[Optional[Agent]] = [None] * self.V_size
    
    def get_h_value(self, config: Config) -> int:
        """Tính giá trị heuristic cho configuration"""
        if self.objective == Objective.OBJ_MAKESPAN:
            return max(self.D.get(i, config[i]) for i in range(self.N))
        elif self.objective == Objective.OBJ_SUM_OF_LOSS:
            return sum(self.D.get(i, config[i]) for i in range(self.N))
        return 0
    
    def get_edge_cost(self, C1: Config, C2: Config) -> int:
        """Lấy chi phí cạnh giữa hai configuration"""
        if self.objective == Objective.OBJ_NONE:
            return 0
        elif self.objective == Objective.OBJ_MAKESPAN:
            return 1
        elif self.objective == Objective.OBJ_SUM_OF_LOSS:
            return sum(1 for i in range(self.N) if C1[i].id != C2[i].id)
        return 0
    
    def funcPIBT(self, ai: Agent) -> bool:
        """
        Thuật toán PIBT cho một agent
        
        Tham số:
            ai: Agent cần lập kế hoạch
        
        Trả về:
            True nếu thành công
        """
        if ai.v_next is not None:
            return True
        
        # Lấy đỉnh hiện tại
        v_now = ai.v_now
        if v_now is None:
            return False
        
        # Các đỉnh ứng viên 
        candidates = [v_now] + v_now.neighbor
        
        # Tính độ ưu tiên dựa trên khoảng cách đến goal
        priorities = []
        for v in candidates:
            dist = self.D.get(ai.id, v)
            tie_breaker = self.tie_breakers[ai.id]
            priorities.append((dist, tie_breaker, v))
        
        priorities.sort()
        
        # Thử từng ứng viên
        for _, _, v in priorities:
            if self.occupied_next[v.id] is not None:
                continue
            
            aj = self.occupied_now[v.id]
            if aj is not None and aj.id != ai.id:
                if not self.funcPIBT(aj):
                    continue
                if aj.v_next is not None and aj.v_next.id == v.id:
                    continue
            
            ai.v_next = v
            self.occupied_next[v.id] = ai
            return True
        
        return False
    
    def get_new_config(self, H: HNode, L: LNode) -> bool:
        """
        Get new configuration using PIBT
        
        Args:
            H: High-level node
            L: Low-level node
        
        Returns:
            True if successful
        """
        # Initialize
        for i in range(self.N):
            self.A[i].v_now = H.C[i]
            self.A[i].v_next = None
            self.occupied_now[H.C[i].id] = self.A[i]
            self.tie_breakers[i] = get_random_float(self.MT)
        
        for v in self.occupied_next:
            v = None
        
        # Đặt các agent cố định từ L
        for i, v in zip(L.who, L.where):
            self.A[i].v_next = v
            self.occupied_next[v.id] = self.A[i]
        
        # Xử lý các agent còn lại
        order = list(range(self.N))
        self.MT.shuffle(order)
        
        for i in order:
            if not self.funcPIBT(self.A[i]):
                for v_id in range(self.V_size):
                    self.occupied_now[v_id] = None
                return False
        
        for v_id in range(self.V_size):
            self.occupied_now[v_id] = None
        
        return True
    
    def expand_lowlevel_tree(self, H: HNode, L: LNode):
        """Mở rộng cây tìm kiếm low-level"""
        # Thử lấy configuration mới với partial assignment hiện tại
        if not self.get_new_config(H, L):
            return
        
        # Lấy configuration mới
        new_config = [self.A[i].v_next for i in range(self.N)]
        
        # Kiểm tra xem đã đến goal chưa
        if is_same_config(new_config, self.ins.goals):  
            pass
        
        if L.depth < self.N:
            for i in range(self.N):
                if i not in L.who:
                    v_now = H.C[i]
                    for v_next in [v_now] + v_now.neighbor:
                        new_L = LNode(L, i, v_next)
                        H.search_tree.append(new_L)
    
    def solve(self, additional_info: str = "") -> Solution:
        """
        Main solving function - implements LaCAM* algorithm
        
        Args:
            additional_info: Additional info string
        
        Returns:
            Solution if found, empty list otherwise
        """
        from .utils import info, is_expired
        
        info(1, self.verbose, "start search")
        
        HNode.HNODE_CNT = 0
        
        # Setup agents
        for i in range(self.N):
            self.A[i] = Agent(i)
        
        # Setup search
        OPEN = []  
        EXPLORED = {} 
        
        h_init = self.get_h_value(self.ins.starts)
        H_init = HNode(self.ins.starts, self.D, None, 0, h_init)
        H_init.search_tree.append(LNode())
        OPEN.append(H_init)
        EXPLORED[self._config_to_tuple(H_init.C)] = H_init
        
        solution = []
        H_goal = None
        
        # DFS
        loop_limit = 100000  # Tăng limit
        while OPEN and not is_expired(self.deadline) and self.loop_cnt < loop_limit:
            self.loop_cnt += 1
            
            H = OPEN[-1]
            
            if self.verbose >= 3:
                print(f"Loop {self.loop_cnt}: |OPEN|={len(OPEN)}, |search_tree|={len(H.search_tree)}, f={H.f}")
            
            if not H.search_tree:
                OPEN.pop()
                continue
            
            if H_goal is not None and H.f >= H_goal.f:
                OPEN.pop()
                continue
            
            if H_goal is None and is_same_config(H.C, self.ins.goals):
                H_goal = H
                info(1, self.verbose, f"found solution, cost: {H.g}")
                if self.objective == Objective.OBJ_NONE:
                    break
                continue
            
            L = H.search_tree.popleft()
            self.expand_lowlevel_tree(H, L)
            
            res = self.get_new_config(H, L)
            if not res:
                continue
            
            C_new = [a.v_next for a in self.A]
            C_new_tuple = self._config_to_tuple(C_new)
            if C_new_tuple in EXPLORED:
                self.rewrite(H, EXPLORED[C_new_tuple], H_goal, OPEN)
                if self.MT is not None and self.MT.random() >= self.RESTART_RATE:
                    H_insert = EXPLORED[C_new_tuple]
                else:
                    H_insert = H_init
                if H_goal is None or H_insert.f < H_goal.f:
                    OPEN.append(H_insert)
            else:
                g_new = H.g + self.get_edge_cost(H.C, C_new)
                h_new = self.get_h_value(C_new)
                H_new = HNode(C_new, self.D, H, g_new, h_new)
                EXPLORED[C_new_tuple] = H_new
                if H_goal is None or H_new.f < H_goal.f:
                    OPEN.append(H_new)
        
        # Save solution
        if H_goal is not None:
            H = H_goal
            while H is not None:
                solution.append(H.C)
                H = H.parent
            solution.reverse()
        
        if H_goal is not None and not OPEN:
            info(1, self.verbose, "solved optimally")
        elif H_goal is not None:
            info(1, self.verbose, "solved sub-optimally")
        elif not OPEN:
            info(1, self.verbose, "no solution")
        else:
            info(1, self.verbose, "timeout")
        
        return solution
    
    def _config_to_tuple(self, config: Config) -> tuple:
        """Convert config to hashable tuple"""
        return tuple(v.id for v in config)
    
    def rewrite(self, H_from: HNode, H_to: HNode, H_goal: Optional[HNode], OPEN: List[HNode]):
        """Rewrite tree structure for optimization"""
        from .utils import info
        from collections import deque
        
        H_from.neighbor.add(H_to)
        
        # Dijkstra update
        Q = deque([H_from])
        while Q:
            n_from = Q.popleft()
            for n_to in n_from.neighbor:
                g_val = n_from.g + self.get_edge_cost(n_from.C, n_to.C)
                if g_val < n_to.g:
                    if n_to == H_goal:
                        info(1, self.verbose, f"cost update: {n_to.g} -> {g_val}")
                    n_to.g = g_val
                    n_to.f = n_to.g + n_to.h
                    n_to.parent = n_from
                    Q.append(n_to)
                    if H_goal is not None and n_to.f < H_goal.f:
                        OPEN.append(n_to)
    
    def get_edge_cost(self, C1: Config, C2: Config) -> int:
        """Get edge cost between two configurations"""
        if self.objective == Objective.OBJ_SUM_OF_LOSS:
            cost = 0
            for i in range(self.N):
                if C1[i] != self.ins.goals[i] or C2[i] != self.ins.goals[i]:
                    cost += 1
            return cost
        return 1
    
    def get_h_value(self, config: Config) -> int:
        """Get heuristic value for a configuration"""
        cost = 0
        if self.objective == Objective.OBJ_MAKESPAN:
            for i in range(self.N):
                cost = max(cost, self.D.get(i, config[i]))
        elif self.objective == Objective.OBJ_SUM_OF_LOSS:
            for i in range(self.N):
                cost += self.D.get(i, config[i])
        return cost
    
    def expand_lowlevel_tree(self, H: HNode, L: LNode):
        """Expand low-level tree"""
        if L.depth >= self.N:
            return
        
        i = H.order[L.depth]
        C = list(H.C[i].neighbor)
        C.append(H.C[i])
        
        if self.MT is not None:
            self.MT.shuffle(C)
        
        for v in C:
            H.search_tree.append(LNode(L, i, v))
    
    def get_new_config(self, H: HNode, L: LNode) -> bool:
        """Get new configuration from low-level node"""
        for a in self.A:
            if a.v_now is not None and self.occupied_now[a.v_now.id] == a:
                self.occupied_now[a.v_now.id] = None
            if a.v_next is not None:
                self.occupied_next[a.v_next.id] = None
                a.v_next = None
            
            a.v_now = H.C[a.id]
            self.occupied_now[a.v_now.id] = a
        
        for k in range(L.depth):
            i = L.who[k]  # agent
            l = L.where[k].id  
            
            if self.occupied_next[l] is not None:
                return False
            
            l_pre = H.C[i].id
            if (self.occupied_next[l_pre] is not None and 
                self.occupied_now[l] is not None and
                self.occupied_next[l_pre].id == self.occupied_now[l].id):
                return False
            
            self.A[i].v_next = L.where[k]
            self.occupied_next[l] = self.A[i]
        
        for k in H.order:
            a = self.A[k]
            if a.v_next is None and not self.funcPIBT(a):
                return False
        
        return True
    
    def funcPIBT(self, ai: Agent) -> bool:
        """Hàm PIBT cho một agent"""
        i = ai.id
        K = len(ai.v_now.neighbor)
        
        self.C_next[i] = []
        for k in range(K):
            u = ai.v_now.neighbor[k]
            self.C_next[i].append(u)
            if self.MT is not None:
                self.tie_breakers[u.id] = self.MT.random()
        self.C_next[i].append(ai.v_now)
        
        self.C_next[i].sort(
            key=lambda v: self.D.get(i, v) + self.tie_breakers[v.id]
        )
        
        swap_agent = self.swap_possible_and_required(ai)
        if swap_agent is not None:
            self.C_next[i].reverse()
        
        for k in range(K + 1):
            u = self.C_next[i][k]
            if self.occupied_next[u.id] is not None:
                continue
            
            ak = self.occupied_now[u.id]
            if ak is not None and ak.v_next == ai.v_now:
                continue
            
            self.occupied_next[u.id] = ai
            ai.v_next = u
            
            if ak is not None and ak != ai and ak.v_next is None:
                if not self.funcPIBT(ak):
                    continue
            
            if (k == 0 and swap_agent is not None and 
                swap_agent.v_next is None and
                self.occupied_next[ai.v_now.id] is None):
                swap_agent.v_next = ai.v_now
                self.occupied_next[swap_agent.v_next.id] = swap_agent
            
            return True
        
        self.occupied_next[ai.v_now.id] = ai
        ai.v_next = ai.v_now
        return False
    
    def swap_possible_and_required(self, ai: Agent) -> Optional[Agent]:
        """Kiểm tra xem swap có khả thi và cần thiết không"""
        i = ai.id
        
        if self.C_next[i][0] == ai.v_now:
            return None
        
        aj = self.occupied_now[self.C_next[i][0].id]
        if (aj is not None and aj.v_next is None and
            self.is_swap_required(ai.id, aj.id, ai.v_now, aj.v_now) and
            self.is_swap_possible(aj.v_now, ai.v_now)):
            return aj
        
        for u in ai.v_now.neighbor:
            ak = self.occupied_now[u.id]
            if ak is None or self.C_next[i][0] == ak.v_now:
                continue
            if (self.is_swap_required(ak.id, ai.id, ai.v_now, self.C_next[i][0]) and
                self.is_swap_possible(self.C_next[i][0], ai.v_now)):
                return ak
        
        return None
    
    def is_swap_required(self, pusher: int, puller: int, 
                        v_pusher_origin: Vertex, v_puller_origin: Vertex) -> bool:
        """Mô phỏng xem swap có cần thiết không"""
        v_pusher = v_pusher_origin
        v_puller = v_puller_origin
        tmp = None
        
        while self.D.get(pusher, v_puller) < self.D.get(pusher, v_pusher):
            n = len(v_puller.neighbor)
            # Loại bỏ các agent không cần di chuyển
            for u in v_puller.neighbor:
                a = self.occupied_now[u.id]
                if (u == v_pusher or
                    (len(u.neighbor) == 1 and a is not None and 
                     self.ins.goals[a.id] == u)):
                    n -= 1
                else:
                    tmp = u
            
            if n >= 2:
                return False  
            if n <= 0:
                break
            
            v_pusher = v_puller
            v_puller = tmp
        
        return (self.D.get(puller, v_pusher) < self.D.get(puller, v_puller) and
                (self.D.get(pusher, v_pusher) == 0 or
                 self.D.get(pusher, v_puller) < self.D.get(pusher, v_pusher)))
    
    def is_swap_possible(self, v_pusher_origin: Vertex, v_puller_origin: Vertex) -> bool:
        """Mô phỏng xem swap có khả thi không"""
        v_pusher = v_pusher_origin
        v_puller = v_puller_origin
        tmp = None
        
        while v_puller != v_pusher_origin:  
            n = len(v_puller.neighbor)  
            for u in v_puller.neighbor:
                a = self.occupied_now[u.id]
                if (u == v_pusher or
                    (len(u.neighbor) == 1 and a is not None and 
                     self.ins.goals[a.id] == u)):
                    n -= 1  
                else:
                    tmp = u  
            
            if n >= 2:
                return True 
            if n <= 0:
                return False
            
            v_pusher = v_puller
            v_puller = tmp
        
        return False
