"""
Xử lý hậu kỳ cho LaCAM2 - kiểm tra tính hợp lệ và metrics của solution
"""

from typing import List
from .instance import Instance, Solution
from .dist_table import DistTable
from .graph import is_same_config
from .utils import info


def is_feasible_solution(ins: Instance, solution: Solution, verbose: int = 0) -> bool:
    """
    Kiểm tra xem solution có hợp lệ không
    
    Tham số:
        ins: Instance của bài toán
        solution: Solution cần kiểm tra
        verbose: Mức độ chi tiết
    
    Trả về:
        True nếu hợp lệ
    """
    if not solution:
        info(1, verbose, "empty solution")
        return False
    
    # Kiểm tra cấu hình bắt đầu
    if not is_same_config(solution[0], ins.starts):
        info(1, verbose, "invalid start configuration")
        return False
    
    # Kiểm tra cấu hình đích
    if not is_same_config(solution[-1], ins.goals):
        info(1, verbose, "invalid goal configuration")
        return False
    
    # Kiểm tra các bước chuyển tiếp và va chạm
    for t in range(len(solution) - 1):
        config_t = solution[t]
        config_t1 = solution[t + 1]
        
        # Kiểm tra va chạm đỉnh (vertex collision)
        locations = set()
        for i in range(ins.N):
            v = config_t1[i]
            if v.id in locations:
                info(1, verbose, f"vertex collision at timestep {t + 1}")
                return False
            locations.add(v.id)
        
        # Kiểm tra tính hợp lệ của bước di chuyển và va chạm cạnh
        for i in range(ins.N):
            v_from = config_t[i]
            v_to = config_t1[i]
            
            # Kiểm tra xem bước di chuyển có hợp lệ không (đứng yên hoặc di chuyển sang ô kề)
            if v_from.id != v_to.id:
                if v_to not in v_from.neighbor:
                    info(1, verbose, f"invalid transition for agent {i} at timestep {t}")
                    return False
        
        # Kiểm tra va chạm hoán đổi (swap collision)
        for i in range(ins.N):
            for j in range(i + 1, ins.N):
                if (config_t[i].id == config_t1[j].id and 
                    config_t[j].id == config_t1[i].id):
                    info(1, verbose, f"swap collision between agents {i} and {j} at timestep {t}")
                    return False
    
    return True


def get_makespan(solution: Solution) -> int:
    """Lấy makespan của solution"""
    return len(solution) - 1


def get_path_cost(solution: Solution, agent_id: int) -> int:
    """Lấy chi phí đường đi cho một agent"""
    cost = 0
    for t in range(len(solution) - 1):
        if solution[t][agent_id].id != solution[t + 1][agent_id].id:
            cost += 1
    return cost


def get_sum_of_costs(solution: Solution) -> int:
    """Lấy tổng chi phí"""
    if not solution:
        return 0
    N = len(solution[0])
    return sum(get_path_cost(solution, i) for i in range(N))


def get_sum_of_loss(solution: Solution) -> int:
    """Lấy tổng độ trễ (delays)"""
    return get_sum_of_costs(solution)


def get_makespan_lower_bound(ins: Instance, D: DistTable) -> int:
    """Lấy cận dưới của makespan"""
    return max(D.get(i, ins.starts[i]) for i in range(ins.N))


def get_sum_of_costs_lower_bound(ins: Instance, D: DistTable) -> int:
    """Lấy cận dưới của tổng chi phí"""
    return sum(D.get(i, ins.starts[i]) for i in range(ins.N))


def print_stats(verbose: int, ins: Instance, solution: Solution, comp_time_ms: float):
    """In thống kê của solution"""
    if not solution:
        info(1, verbose, "no solution found")
        return
    
    D = DistTable(ins)
    
    makespan = get_makespan(solution)
    makespan_lb = get_makespan_lower_bound(ins, D)
    makespan_ub = makespan / makespan_lb if makespan_lb > 0 else 0
    
    sum_of_costs = get_sum_of_costs(solution)
    sum_of_costs_lb = get_sum_of_costs_lower_bound(ins, D)
    sum_of_costs_ub = sum_of_costs / sum_of_costs_lb if sum_of_costs_lb > 0 else 0
    
    sum_of_loss = get_sum_of_loss(solution)
    sum_of_loss_lb = sum_of_costs_lb
    sum_of_loss_ub = sum_of_loss / sum_of_loss_lb if sum_of_loss_lb > 0 else 0
    
    info(1, verbose, 
         f"solved: {comp_time_ms:.0f}ms\t"
         f"makespan: {makespan} (lb={makespan_lb}, ub={makespan_ub:.2f})\t"
         f"sum_of_costs: {sum_of_costs} (lb={sum_of_costs_lb}, ub={sum_of_costs_ub:.2f})\t"
         f"sum_of_loss: {sum_of_loss} (lb={sum_of_loss_lb}, ub={sum_of_loss_ub:.2f})")


def make_log(ins: Instance, solution: Solution, output_name: str, 
             comp_time_ms: float, map_name: str, seed: int,
             additional_info: str, log_short: bool = False):
    """
    Ghi solution ra file
    
    Tham số:
        ins: Instance của bài toán
        solution: Solution để ghi
        output_name: Đường dẫn file output
        comp_time_ms: Thời gian tính toán (mili giây)
        map_name: Tên file map
        seed: Random seed
        additional_info: Thông tin bổ sung
        log_short: Nếu True, không ghi đường đi đầy đủ
    """
    try:
        with open(output_name, 'w') as file:
            file.write(f"map: {map_name}\n")
            file.write(f"seed: {seed}\n")
            file.write(f"agents: {ins.N}\n")
            file.write(f"comp_time_ms: {comp_time_ms:.2f}\n")
            
            if solution:
                D = DistTable(ins)
                file.write(f"makespan: {get_makespan(solution)}\n")
                file.write(f"makespan_lower_bound: {get_makespan_lower_bound(ins, D)}\n")
                file.write(f"sum_of_costs: {get_sum_of_costs(solution)}\n")
                file.write(f"sum_of_costs_lower_bound: {get_sum_of_costs_lower_bound(ins, D)}\n")
                
                if not log_short:
                    file.write("\npaths:\n")
                    for i in range(ins.N):
                        path = [solution[t][i].index for t in range(len(solution))]
                        file.write(f"{i}: {' '.join(map(str, path))}\n")
            else:
                file.write("status: failed\n")
            
            if additional_info:
                file.write(f"\n{additional_info}\n")
                
    except IOError as e:
        print(f"Error writing to file {output_name}: {e}")
