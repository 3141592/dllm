import torch
vector = torch.tensor([2., 3., 4.])
print("Vector: ", vector)

print("torch.norm(vector)", torch.norm(vector))

tensor_int64  = torch.tensor([1, 2, 3], dtype=torch.int64)
print("tensor_int64: ", tensor_int64)

tensor_float = tensor_int64.to(torch.float64)
print("tensor_float: ", tensor_float)



