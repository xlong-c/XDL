import torch


A = torch.randn(3, 5)
B = torch.randn(5, 4)

C = torch.mm(A, B)
print("torch.mm(A, B):")
print(C)

rows_a, cols_a = A.shape
rows_b, cols_b = B.shape
assert cols_a == rows_b

C_row_col = torch.zeros(rows_a, cols_b, dtype=C.dtype)
for i in range(rows_a):
    a_row = A[i, :]
    for j in range(cols_b):
        b_col = B[:, j]
        C_row_col[i, j] = torch.dot(a_row, b_col)

print("row x col:")
print(C_row_col)
print("allclose:", torch.allclose(C, C_row_col))

C_col_row = torch.zeros(rows_a, cols_b, dtype=C.dtype)
for k in range(cols_a):
    a_col = A[:, k]
    b_row = B[k, :]
    C_col_row += torch.outer(a_col, b_row)
    print(torch.outer(a_col, b_row).shape)

print("col x row sum:")
print(C_col_row)
print("allclose:", torch.allclose(C, C_col_row))
