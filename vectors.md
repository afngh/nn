# MiniFramework - Vectors and Matrices Library

A lightweight, dependency-free Python library for performing fundamental vector and matrix operations. The library utilizes standard Python lists (`list[float]` and `list[list[float]]`) to represent mathematical vectors and matrices, making it highly portable and easy to integrate.

---

## Table of Contents
- [Vector Operations](#vector-operations)
  - [add](#add)
  - [sub](#sub)
  - [sv_mul (Vector)](#sv_mul-vector)
  - [dot](#dot)
  - [elewise_mul](#elewise_mul)
  - [vector_sum](#vector_sum)
- [Matrix Operations](#matrix-operations)
  - [mat_vec_mul](#mat_vec_mul)
  - [mat_mul](#mat_mul)
  - [add_mat](#add_mat)
  - [sub_mat](#sub_mat)
  - [sv_mul (Matrix)](#sv_mul-matrix)
- [Important Note: Name Collision / Bug Alert](#important-note-name-collision--bug-alert)
- [Exception Handling](#exception-handling)

---

## Vector Operations

### `add`
Adds two vectors element-wise.
```python
def add(v1: list[float], v2: list[float]) -> list[float]
```
- **Parameters:**
  - `v1`: The first input vector.
  - `v2`: The second input vector.
- **Returns:** A new vector representing the element-wise sum $\vec{v_1} + \vec{v_2}$.
- **Raises:** `ValueError` if the vectors are not of the same length.

### `sub`
Subtracts the second vector from the first vector element-wise.
```python
def sub(v1: list[float], v2: list[float]) -> list[float]
```
- **Parameters:**
  - `v1`: The vector to subtract from.
  - `v2`: The vector to subtract.
- **Returns:** A new vector representing the element-wise difference $\vec{v_1} - \vec{v_2}$.
- **Raises:** `ValueError` if the vectors are not of the same length.

### `sv_mul` (Vector)
Multiplies a vector by a scalar value.
```python
def sv_mul(scalar: float, v: list[float]) -> list[float]
```
- **Parameters:**
  - `scalar`: The scalar multiplier.
  - `v`: The input vector.
- **Returns:** A new vector where every element of `v` is multiplied by `scalar`.
- **Note:** *See the [Name Collision](#important-note-name-collision--bug-alert) section below regarding this function.*

### `dot`
Calculates the dot (scalar) product of two vectors.
```python
def dot(v1: list[float], v2: list[float]) -> float
```
- **Parameters:**
  - `v1`: The first vector.
  - `v2`: The second vector.
- **Returns:** The sum of the element-wise products: $\sum (v1_i \times v2_i)$.
- **Raises:** `ValueError` if the vectors are not of the same length.

### `elewise_mul`
Performs element-wise multiplication (Hadamard product) on two vectors.
```python
def elewise_mul(v1: list[float], v2: list[float]) -> list[float]
```
- **Parameters:**
  - `v1`: The first vector.
  - `v2`: The second vector.
- **Returns:** A new vector representing the element-wise product of `v1` and `v2`.
- **Raises:** `ValueError` if the vectors are not of the same length.

### `vector_sum`
Computes the sum of all elements in a single vector.
```python
def vector_sum(v: list[float]) -> float
```
- **Parameters:**
  - `v`: The input vector.
- **Returns:** The sum of all elements in `v`.

---

## Matrix Operations

### `mat_vec_mul`
Multiplies a matrix by a vector.
```python
def mat_vec_mul(matrix: list[list[float]], v: list[float]) -> list[float]
```
- **Parameters:**
  - `matrix`: A 2D list representing the matrix.
  - `v`: A list representing the vector.
- **Returns:** A new vector representing the product of the matrix and the vector.
- **Raises:** `ValueError` if any row in the matrix does not match the length of the vector.

### `mat_mul`
Performs matrix multiplication on two 2D matrices.
```python
def mat_mul(m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]
```
- **Parameters:**
  - `m1`: The first matrix.
  - `m2`: The second matrix.
- **Returns:** A new 2D matrix representing the matrix product $M_1 \times M_2$.
- **Raises:** `ValueError` if the number of columns in `m1` does not match the number of rows in `m2`.

### `add_mat`
Adds two matrices of the same dimensions element-wise.
```python
def add_mat(m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]
```
- **Parameters:**
  - `m1`: The first matrix.
  - `m2`: The second matrix.
- **Returns:** A new 2D matrix representing the element-wise sum of the two matrices.

### `sub_mat`
Subtracts the second matrix from the first matrix of the same dimensions element-wise.
```python
def sub_mat(m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]
```
- **Parameters:**
  - `m1`: The matrix to subtract from.
  - `m2`: The matrix to subtract.
- **Returns:** A new 2D matrix representing the element-wise difference of the two matrices.

### `sv_mul` (Matrix)
Multiplies a matrix by a scalar value.
```python
def sv_mul(scalar: float, m: list[list[float]]) -> list[list[float]]
```
- **Parameters:**
  - `scalar`: The scalar multiplier.
  - `m`: The input matrix.
- **Returns:** A new 2D matrix where every element of `m` is multiplied by `scalar`.

---

## Important Note: Name Collision / Bug Alert

> [!WARNING]
> **Duplicate Function Name Conflict**
> In the current implementation of `vectors.py`, the function `sv_mul` is defined twice:
> 1. For scalar-vector multiplication (lines 17–18).
> 2. For scalar-matrix multiplication (lines 45–46).
>
> In Python, defining a function with the same name twice in the same scope overrides the previous definition. Consequently, calling `sv_mul(scalar, vector)` with a 1D vector will fail or result in incorrect behavior because only the **scalar-matrix** version (the last one defined) is active.
>
> **Recommended Fix:**
> Rename the scalar-vector multiplication function to `sv_mul` and the scalar-matrix multiplication function to `sm_mul` (scalar-matrix multiply).

---

## Exception Handling

### `LengthError`
A helper utility function that raises a standard error when inputs do not have matching dimensions.
```python
def LengthError()
```
- **Raises:** `ValueError("Vectors must be of the same length")`
