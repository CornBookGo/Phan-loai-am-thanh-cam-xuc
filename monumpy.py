import numpy as np
import matplotlib.pyplot as plt

# Đọc file .npy
data = np.load("features/test_mfcc-chroma-mel_HNS_488.npy")

# Kiểm tra thông tin cơ bản
print("Dạng:", data.shape)
print("Dạng dữ liệu:", data.dtype)
print("Phần tử đầu tiên:", data[1])  # Nếu là mảng nhiều phần tử

