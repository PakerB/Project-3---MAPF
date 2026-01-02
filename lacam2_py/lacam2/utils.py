"""
Các hàm tiện ích cho LaCAM2
"""

import time
import random
from typing import Optional


def info(level: int, verbose: int, *args) -> None:
    """In thông tin dựa trên mức độ chi tiết"""
    if verbose >= level:
        print(*args)


class Deadline:
    """Quản lý thời gian để kiểm tra deadline"""
    
    def __init__(self, time_limit_ms: float = 0):
        self.t_start = time.perf_counter()
        self.time_limit_ms = time_limit_ms
    
    def elapsed_ms(self) -> float:
        """Lấy thời gian đã trôi qua tính bằng mili giây"""
        return (time.perf_counter() - self.t_start) * 1000
    
    def elapsed_ns(self) -> float:
        """Lấy thời gian đã trôi qua tính bằng nano giây"""
        return (time.perf_counter() - self.t_start) * 1_000_000_000
    

def elapsed_ms(deadline: Optional[Deadline]) -> float:
    """Lấy thời gian đã trôi qua từ deadline"""
    if deadline is None:
        return 0
    return deadline.elapsed_ms()


def elapsed_ns(deadline: Optional[Deadline]) -> float:
    """Lấy thời gian đã trôi qua từ deadline tính bằng nano giây"""
    if deadline is None:
        return 0
    return deadline.elapsed_ns()


def is_expired(deadline: Optional[Deadline]) -> bool:
    """Kiểm tra xem deadline đã hết hạn chưa"""
    if deadline is None:
        return False
    return deadline.elapsed_ms() > deadline.time_limit_ms


def get_random_float(mt: random.Random, from_val: float = 0, to_val: float = 1) -> float:
    """Lấy số thực ngẫu nhiên trong khoảng"""
    return mt.uniform(from_val, to_val)


def get_random_int(mt: random.Random, from_val: int = 0, to_val: int = 1) -> int:
    """Lấy số nguyên ngẫu nhiên trong khoảng (bao gồm cả 2 đầu)"""
    return mt.randint(from_val, to_val)
