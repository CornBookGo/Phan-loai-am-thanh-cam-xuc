"""
Lệnh tìm kiếm dạng lưới tất cả các tham số cung cấp trong parameters.py bao gồm phân loại (classifiers) và hồi quy (regressors).
Lưu ý: thực thi lệnh có thể mất hàng giờ để tìm kiếm tham số mô hình tốt nhất cho các thuật toán khác nhau, loại bỏ 1 số tham số để tìm kiếm nhanh hơn.
"""

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
        # Trường hợp thuật toán K-Neighbor gần nhất
        # đặt số lượng neighbor theo chiều dài của cảm xúc
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} dat {cv_best_score:.3f} diem chinh xac xac nhan (validation)")

print(f"Lua chon bo phan loai tot nhat cho {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_classifiers.pickle", "wb"))

best_estimators = []

for model, params in regression_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsRegressor":
        # Trường hợp thuật toán K-Neighbor gần nhất
        # đặt số lượng neighbor theo chiều dài của cảm xúc
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions, classification=False)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} dat {cv_best_score:.3f} trong tong diem sai so tuyet doi trung binh!")

print(f"Lua chon bo hoi quy tot nhat cho {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_regressors.pickle", "wb"))



# SVC: C=0.001, gamma=0.001, kernel='poly'
# AdaBoostClassifier: {'algorithm': 'SAMME', 'learning_rate': 0.8, 'n_estimators': 60}
# RandomForestClassifier: {'max_depth': 7, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 40}
# GradientBoostingClassifier: {'learning_rate': 0.3, 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 70, 'subsample': 0.7}
# DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
# KNeighborsClassifier: {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
# MLPClassifier: {'alpha': 0.005, 'batch_size': 256, 'hidden_layer_sizes': (300,), 'learning_rate': 'adaptive', 'max_iter': 500}