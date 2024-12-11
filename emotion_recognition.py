from data_extractor import load_data
from utils import extract_feature, AVAILABLE_EMOTIONS
from create_csv import write_emodb_csv, write_tess_ravdess_csv, write_custom_csv

from sklearn.metrics import accuracy_score, make_scorer, fbeta_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as pl
from time import time
from utils import get_best_estimators, get_audio_config
import numpy as np
import tqdm
import os
import random
import pandas as pd


class EmotionRecognizer:
    """Lop cho dao tao, kiem tra va du doan cam xuc dua tren dac diem cua tieng noi trich xuat va dua vao mo hinh `sklearn` hoac `keras`"""
    def __init__(self, model=None, **kwargs):
        """
        Thong so
            model (sklearn model): mo hinh su dung de nhan dien cam xuc. Neu 'model' la None, thi tu dong goi den
            emotions (list): danh sach cam xuc su dung. Trong do no phai co trong du lieu Ravdess, Tess va Emodb, co 9 loai cam xuc rieng biet gom: 'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps' (surprised), 'boredom'. Mac dinh la ["sad", "neutral", "happy"].
            tess_ravdess (bool): nen dung du lieu Tess va Ravdess khong, mac dinh la True
            emodb (bool): nen dung du lieu Emodb khong, mac dinh la True,
            custom_db (bool): whether to use custom Speech dataset that is located in `data/train-custom`
                and `data/test-custom`, default is True
            tess_ravdess_name (str): the name of the output CSV file for TESS&RAVDESS dataset, default is "tess_ravdess.csv"
            emodb_name (str): the name of the output CSV file for EMO-DB dataset, default is "emodb.csv"
            custom_db_name (str): the name of the output CSV file for the custom dataset, default is "custom.csv"
            features (list): list of speech features to use, default is ["mfcc", "chroma", "mel"]
                (i.e MFCC, Chroma and MEL spectrogram )
            classification (bool): whether to use classification or regression, default is True
            balance (bool): whether to balance the dataset ( both training and testing ), default is True
            verbose (bool/int): whether to print messages on certain tasks, default is 1
        Note that when `tess_ravdess`, `emodb` and `custom_db` are set to `False`, `tess_ravdess` will be set to True
        automatically.
        """
        # cảm xúc
        self.emotions = kwargs.get("emotions", ["sad", "neutral", "happy"])
        # đảm bảo phải là những cảm xúc đã được nhập
        self._verify_emotions()
        # cấu hình của âm thanh
        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"])
        self.audio_config = get_audio_config(self.features)
        # tải dữ liệu
        self.tess_ravdess = kwargs.get("tess_ravdess", True)
        self.emodb = kwargs.get("emodb", True)
        self.custom_db = kwargs.get("custom_db", True)

        if not self.tess_ravdess and not self.emodb and not self.custom_db:
            self.tess_ravdess = True
    
        self.classification = kwargs.get("classification", True)
        self.balance = kwargs.get("balance", True)
        self.override_csv = kwargs.get("override_csv", True)
        self.verbose = kwargs.get("verbose", 1)

        self.tess_ravdess_name = kwargs.get("tess_ravdess_name", "tess_ravdess.csv")
        self.emodb_name = kwargs.get("emodb_name", "emodb.csv")
        self.custom_db_name = kwargs.get("custom_db_name", "custom.csv")

        self.verbose = kwargs.get("verbose", 1)

        # đặt tên đường dẫn CSV
        self._set_metadata_filenames()
        # và viết dữ liệu vào nó
        self.write_csv()

        # đặt thuộc tính Boolean (True hoặc False)
        self.data_loaded = False
        self.model_trained = False

        # mô hình
        if not model:
            self.determine_best_model()
        else:
            self.model = model

    def _set_metadata_filenames(self):
        """
        Phuong phap bao ve dua tat ca tep CSV vao 2 phien ban dac trung:
        - `self.train_desc_files` cho CSV dao tao
        - `self.test_desc_files` cho CSV kiem tra
        """
        train_desc_files, test_desc_files = [], []
        if self.tess_ravdess:
            train_desc_files.append(f"train_{self.tess_ravdess_name}")
            test_desc_files.append(f"test_{self.tess_ravdess_name}")
        if self.emodb:
            train_desc_files.append(f"train_{self.emodb_name}")
            test_desc_files.append(f"test_{self.emodb_name}")
        if self.custom_db:
            train_desc_files.append(f"train_{self.custom_db_name}")
            test_desc_files.append(f"test_{self.custom_db_name}")

        # đặt tất cả tệp đã tạo thành thuộc tính đối tượng (Object)
        self.train_desc_files = train_desc_files
        self.test_desc_files  = test_desc_files

    def _verify_emotions(self):
        """
        Dam bao cam xuc truyen vao hop le
        """
        for emotion in self.emotions:
            assert emotion in AVAILABLE_EMOTIONS, "Cam xuc khong duoc chap nhan"

    def get_best_estimators(self):
        """Tai cac cong cu uoc tinh tu tep 'grid' va tra ve gia tri"""
        return get_best_estimators(self.classification)

    def write_csv(self):
        """
        Viet cac tep CSV co san trong `self.train_desc_files` va `self.test_desc_files` duoc xac thuc bang phuong thuc `self._set_metadata_filenames()`
        """
        for train_csv_file, test_csv_file in zip(self.train_desc_files, self.test_desc_files):
            # tiếp cận không an toàn
            if os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
                # tệp đã có, không cần phải ghi vào CSV
                if not self.override_csv:
                    continue
            if self.emodb_name in train_csv_file:
                write_emodb_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("Da tao tep CSV Emodb")
            elif self.tess_ravdess_name in train_csv_file:
                write_tess_ravdess_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("Da tao tep CSV Tess va Ravdess")
            elif self.custom_db_name in train_csv_file:
                write_custom_csv(emotions=self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("Da tao tep CSV CustomDB")

    def load_data(self):
        """
        Tai va trich xuat dac trung tu tep am thanh cho du lieu duoc chinh dinh
        """
        if not self.data_loaded:
            result = load_data(self.train_desc_files, self.test_desc_files, self.audio_config, self.classification,
                                emotions=self.emotions, balance=self.balance)
            self.X_train = result['X_train']
            self.X_test = result['X_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.train_audio_paths = result['train_audio_paths']
            self.test_audio_paths = result['test_audio_paths']
            self.balance = result["balance"]
            if self.verbose:
                print("Du lieu da duoc tai len")
            self.data_loaded = True

    def train(self, verbose=1):
        """
        Huan luyen mo hinh, neu du lieu chua duoc tai thi se tu dong tai len
        """
        if not self.data_loaded:
            # neu du lieu chua duoc tai thi tai no sau
            self.load_data()
        if not self.model_trained:
            self.model.fit(X=self.X_train, y=self.y_train)
            self.model_trained = True
            if verbose:
                print("Mo hinh da duoc dao tao")

    def predict(self, audio_path):
        """
        Dua 'audio_path' vao, de trich xuat cac dac trung va du doan cam xuc
        """
        feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
        return self.model.predict(feature)[0]

    def predict_proba(self, audio_path):
        """
        Du doan xac suat tung cam xuc
        """
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Du doan xac suat khong co y nghia voi hoi quy")

    def grid_search(self, params, n_jobs=2, verbose=1):
        """
        Thuc hien GridSearchCV tren 'params' truyen vao 'self.model'
        Tra ve bo danh gia: (best_estimator, best_params, best_score).
        """
        score = accuracy_score if self.classification else mean_absolute_error
        grid = GridSearchCV(estimator=self.model, param_grid=params, scoring=make_scorer(score), n_jobs=n_jobs, verbose=verbose, cv=3)
        grid_result = grid.fit(self.X_train, self.y_train)
        return grid_result.best_estimator_, grid_result.best_params_, grid_result.best_score_

    def determine_best_model(self):
        """
        Tai cac cong cu uoc tinh tot nhat va xac dinh cho du lieu thu nghiem, sau do dat no thanh `self.model`.
        Trong hoi quy, so lieu duoc su dung la MSE va do chinh xac de phan loai.
        Luu y: thuc hien co the mat vai phut do de dao tao cac cong cu uoc tinh (luu trong thu muc 'grid') de xac dinh cong cu uoc tinh tot nhat.
        """
        if not self.data_loaded:
            self.load_data()
        
        # tải các công cụ (mfcc - chroma - mel)
        estimators = self.get_best_estimators()

        result = []

        if self.verbose:
            estimators = tqdm.tqdm(estimators)

        for estimator, params, cv_score in estimators:
            if self.verbose:
                estimators.set_description(f"Danh gia {estimator.__class__.__name__}")
            detector = EmotionRecognizer(estimator, emotions=self.emotions, tess_ravdess=self.tess_ravdess,
                                        emodb=self.emodb, custom_db=self.custom_db, classification=self.classification,
                                        features=self.features, balance=self.balance, override_csv=False)
            # dữ liệu đã được tải
            detector.X_train = self.X_train
            detector.X_test  = self.X_test
            detector.y_train = self.y_train
            detector.y_test  = self.y_test
            detector.data_loaded = True
            # đào tạo mô hình
            detector.train(verbose=0)
            # đưa ra độ chính xác khi kiểm tra
            accuracy = detector.test_score()
            # đưa ra kết quả
            result.append((detector.model, accuracy))

        # sắp xếp kết quả
        # hồi quy: tốt nhất là ở dưới thấ; phân loại: tốt nhất là ở trên cao
        result = sorted(result, key=lambda item: item[1], reverse=self.classification)
        best_estimator = result[0][0]
        accuracy = result[0][1]
        self.model = best_estimator
        self.model_trained = True
        if self.verbose:
            if self.classification:
                print(f"Mo hinh tot nhat duoc xac dinh: {self.model.__class__.__name__} voi {accuracy*100:.3f}% do chinh xac duoc kiem tra")
            else:
                print(f"Mo hinh tot nhat duoc xac dinh: {self.model.__class__.__name__} voi {accuracy:.5f} co nghia la sai so tuyet doi")

    def test_score(self):
        """
        Tinh ket qua trong du lieu kiem tra
        Neu nhu `self.classification` bang True, so lieu duoc su dung la do chinh xac,
        Mean-Squared-Error su dung theo cach khac (hoi quy)
        """
        y_pred = self.model.predict(self.X_test)
        if self.classification:
            return accuracy_score(y_true=self.y_test, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_test, y_pred=y_pred)

    def train_score(self):
        """
        Tinh ket qua trong du lieu dao tao
        Neu nhu `self.classification` bang True, so lieu duoc su dung la do chinh xac,
        Mean-Squared-Error su dung theo cach khac (hoi quy)
        """
        y_pred = self.model.predict(self.X_train)
        if self.classification:
            return accuracy_score(y_true=self.y_train, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_train, y_pred=y_pred)

    def train_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_train)
        return fbeta_score(self.y_train, y_pred, beta, average='micro')

    def test_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_test)
        return fbeta_score(self.y_test, y_pred, beta, average='micro')

    def confusion_matrix(self, percentage=True, labeled=True):
        """
        Tinh ma tran nham lan (confusion matrix) de danh gia do chinh xac khi kiem tra phan loai va tra ve duoi dang khung ma tran numpy hoac khung du lieu pandas
        Thong so:
            percentage (bool): nen dung ty le phan tram thay vi so luong mau, mac dinh la True.
            labeled (bool): co gan nhan cho cot va chỉ mục trong khung du lieu.
        """
        if not self.classification:
            raise NotImplementedError("Ma tran nham lan chi hoat dong khi co van de phan loai")
        y_pred = self.model.predict(self.X_test)
        matrix = confusion_matrix(self.y_test, y_pred, labels=self.emotions).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # chuyển sang tỷ lệ phần trăm
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[ f"true_{e}" for e in self.emotions ],
                                    columns=[ f"predicted_{e}" for e in self.emotions ])
        return matrix

    def draw_confusion_matrix(self):
        """Calculates the confusion matrix and shows it"""
        matrix = self.confusion_matrix(percentage=False, labeled=False)
        # thêm nhãn, tiêu đề, liên kết,...
        pl.imshow(matrix, cmap="binary")
        pl.show()

    def get_n_samples(self, emotion, partition):
        """Tra ve mau du lieu so cua lop 'emotion' trong 'partition' cu the (test hoac train)
        """
        if partition == "test":
            return len([y for y in self.y_test if y == emotion])
        elif partition == "train":
            return len([y for y in self.y_train if y == emotion])

    def get_samples_by_class(self):
        """
        Tra ve khung du lieu chua so luong mau dao tao va kiem tra cho moi cam xuc
        Neu du lieu chua duoc tai se tu dong tai
        """
        if not self.data_loaded:
            self.load_data()
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.emotions:
            n_train = self.get_n_samples(emotion, "train")
            n_test = self.get_n_samples(emotion, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)
        
        # đưa ra tổng số lượng
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(data={"train": train_samples, "test": test_samples, "total": total}, index=self.emotions + ["total"])

    def get_random_emotion(self, emotion, partition="train"):
        """
        Tra ve chi muc du lieu `emotion` ngau nhien tren `partition`.
        """
        if partition == "train":
            index = random.choice(list(range(len(self.y_train))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_train))))
        elif partition == "test":
            index = random.choice(list(range(len(self.y_test))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_test))))
        else:
            raise TypeError("Phan vung khong hop le, chi co 'train' va 'test' duoc chap nhan")

        return index


def plot_histograms(classifiers=True, beta=0.5, n_classes=3, verbose=1):
    """
    Tai cac cong cu uoc tinh khac nhau tu thu muc 'grid' va tinh toan 1 so thong ke de ve bieu do.
    Thong so:
        classifiers (bool): neu `True`, se ve do thi phan, neu khong thi hoi quy
        beta (float): beta value for calculating fbeta score for various estimators.
        n_classes (int): number of classes
    """
    # lấy các công cụ ước tính từ kết quả tìm kiếm lưới được thực hiện
    estimators = get_best_estimators(classifiers)

    final_result = {}
    for estimator, params, cv_score in estimators:
        final_result[estimator.__class__.__name__] = []
        for i in range(3):
            result = {}
            # khởi tạo lớp
            detector = EmotionRecognizer(estimator, verbose=0)
            # tải dữ liệu
            detector.load_data()
            if i == 0:
                # bước đầu đưa 1% dữ liệu mẫu
                sample_size = 0.01
            elif i == 1:
                # bước hai đưa 10% dữ liệu mẫu
                sample_size = 0.1
            elif i == 2:
                # đưa tất cả dữ liệu
                sample_size = 1
            # tính toán số mẫu huấn luyện và mẫu kiểm tra
            n_train_samples = int(len(detector.X_train) * sample_size)
            n_test_samples = int(len(detector.X_test) * sample_size)
            # thiết lập dữ liệu
            detector.X_train = detector.X_train[:n_train_samples]
            detector.X_test = detector.X_test[:n_test_samples]
            detector.y_train = detector.y_train[:n_train_samples]
            detector.y_test = detector.y_test[:n_test_samples]
            # tính toán thời gian đào tạo
            t_train = time()
            detector.train()
            t_train = time() - t_train
            # tính toán thời gian kiểm tra
            t_test = time()
            test_accuracy = detector.test_score()
            t_test = time() - t_test
            # đặt kết quả từ tên tương ứng
            result['train_time'] = t_train
            result['pred_time'] = t_test
            result['acc_train'] = cv_score
            result['acc_test'] = test_accuracy
            result['f_train'] = detector.train_fbeta_score(beta)
            result['f_test'] = detector.test_fbeta_score(beta)
            if verbose:
                print(f" {estimator.__class__.__name__} voi {sample_size*100}% ({n_train_samples}) mau du lieu dat duoc {cv_score*100:.3f}% do Xac thuc trong {t_train:.3f} giay va {test_accuracy*100:.3f}% do Kiem tra trong {t_test:.3f} giay")
            # nối tên với danh sách kết quả
            final_result[estimator.__class__.__name__].append(result)
        if verbose:
            print()
    visualize(final_result, n_classes=n_classes)
    


def visualize(results, n_classes):
    """
    Hien thi ket qua cua nhieu cong cu hoc may khac nhau
    
    Dau vao:
      - results: danh sach ten cong cu chua cac ket qua khac nhau 
      - n_classes: so luong lop
    """

    n_estimators = len(results)

    # dự đoán thực
    accuracy = 1 / n_classes
    f1 = 1 / n_classes
    # khởi tạo khung hình
    fig, ax = pl.subplots(2, 4, figsize = (11,7))
    # khởi tạo hằng số
    bar_width = 0.4
    colors = [ (random.random(), random.random(), random.random()) for _ in range(n_estimators) ]
    # vòng lặp vẽ 4 bảng dữ liệu
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                x = bar_width * n_estimators
                # tạo biểu đồ
                ax[j//3, j%3].bar(i*x+k*(bar_width), results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([x-0.2, x*2-0.2, x*3-0.2])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.2, x*3))
    # thêm trục tung nhãn y
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    # thêm tên gọi
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    # thêm trục hoành trong biểu đồ
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    # đặt giới hạn của y trong bảng
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    # đặt các ô bổ sung trống
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')
    # tạo chú thích
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()
    # Tiêu đề
    pl.suptitle("So lieu hieu suat cho 3 mo hinh hoc giam sat", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    