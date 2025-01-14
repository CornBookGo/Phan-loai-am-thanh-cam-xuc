import pickle

from emotion_recognition import EmotionRecognizer
from parameters import classification_grid_parameters, regression_grid_parameters

# các lớp cảm xúc được đưa vào khi tìm kiếm dạng lưới
emotions = ['sad', 'neutral', 'happy']
# số lượng việc tìm song song khi tìm kiếm lưới
n_jobs = 4

best_estimators = []

for model, params in classification_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsClassifier":
        # Trường hợp thuật toán K-láng giềng gần nhất
        # đặt số lượng neighbor theo số lượng của cảm xúc
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} đạt {cv_best_score:.3f} độ xác nhận (validation)")

print(f"Lựa chọn bộ phân loại tốt nhất cho {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_classifiers.pickle", "wb"))

best_estimators = []

for model, params in regression_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsRegressor":
        # Trường hợp thuật toán K-láng giềng gần nhất
        # đặt số lượng neighbor theo chiều dài của cảm xúc
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions, classification=False)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} đạt {cv_best_score:.3f} tổng điểm sai số tuyệt đối trung bình!")

print(f"Lựa chọn bộ hồi quy tốt nhất cho {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_regressors.pickle", "wb"))
