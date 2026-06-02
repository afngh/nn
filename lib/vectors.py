import math
import random

def LengthError():
    raise ValueError("Vectors must be of the same length")

def add(v1:list[float],v2:list[float]) -> list[float]:
    if len(v1) != len(v2):
        LengthError()
    return [x + y for x,y in zip(v1,v2)]

def sub(v1:list[float],v2:list[float]) -> list[float]:
    if len(v1) != len(v2):
        LengthError()
    return [x - y for x,y in zip(v1,v2)]

def sv_mul(scalar:float,v:list[float]) -> list[float]:
    return [scalar * x for x in v]

def dot(v1:list[float],v2:list[float]) -> float:
    if len(v1) != len(v2):
        LengthError()
    return sum([x*y for x,y in zip(v1,v2)])

def elewise_mul(v1:list[float],v2:list[float]) -> list[float]:
    if len(v1) != len(v2):
        LengthError()
    return [x * y for x,y in zip(v1,v2)]

def vector_sum(v:list[float]) -> float:
    return sum(v)

def mat_vec_mul(matrix:list[list[float]],v:list[float]) -> list[float]:
    return [dot(row,v) for row in matrix]

def mat_mul(m1:list[list[float]],m2:list[list[float]]) -> list[list[float]]:
    return [[dot(r,c) for c in zip(*m2)] for r in m1]

def add_mat(m1:list[list[float]],m2:list[list[float]]) -> list[list[float]]:
    return [[x + y for x,y in zip(row1,row2)] for row1,row2 in zip(m1,m2)]

def sub_mat(m1:list[list[float]],m2:list[list[float]]) -> list[list[float]]:
    return [[x - y for x,y in zip(row1,row2)] for row1,row2 in zip(m1,m2)]

def sv_mul(scalar:float,m:list[list[float]]) -> list[list[float]]:
    return [[scalar * x for x in row] for row in m]