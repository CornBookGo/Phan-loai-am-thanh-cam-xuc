import pickle
import os

folder_name = 'grid'  
file_name = 'best_classifiers.pickle'

file_path = os.path.join(folder_name, file_name)
try:
    with open(file_path, 'rb') as file:
        best_params = pickle.load(file)
        print("Thông số tốt nhất được lưu trong file:")
        print(best_params)
except FileNotFoundError:
    print(f"File không tồn tại tại đường dẫn: {file_path}")
except pickle.UnpicklingError:
    print("Có lỗi xảy ra khi đọc file pickle.")
