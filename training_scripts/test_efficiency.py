import torch


a = torch.rand(100000)

a = a + 1


# what we need to compare:
# 1. square -> square root vs only square
# 2. exp vs inverse
import time

start = time.time()
b = a ** 2
b = b ** 0.5
print(time.time() - start)

start = time.time()
b = a ** 2
print(time.time() - start)

start = time.time()
b = torch.exp(a)

print(time.time() - start)

start = time.time()
b = 1 / a

print(time.time() - start)