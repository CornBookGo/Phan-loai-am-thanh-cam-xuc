import numpy as np
import pandas as pd
import pickle
import tqdm
import os

from utils import get_label, extract_feature, get_first_letters
from collections import defaultdict

# Mô tả các đoạn âm thanh và cung cấp cho các thuật toán học máy để đào tạo và kiểm tra
class AudioExtractor:
    def __init__(self, audio_config=None, verbose=1, features_folder_name="features", classification=True, emotions=['sad', 'neutral', 'happy'], balance=True):
        self.audio_config = audio_config if audio_config else {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False}
        self.verbose = verbose
        self.features_folder_name = features_folder_name
        self.classification = classification
        self.emotions = emotions
        self.balance = balance
        # Kích thước đầu vào
        self.input_dimension = None
        """
        Thong so:
            audio_config (dict): luu tru de biet nhung tinh nang nao can trich xuat tu tep am thanh,mac dinh la {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False} (la: mfcc, chroma and mel)
            verbose (bool/int): muc do tu ngu, 0 cho im lang, 1 cho thong tin, mac dinh la 1
            features_folder_name (str): thu muc luu cac tinh nang dau ra duoc trich xuat, mac dinh la "features".
            classification (bool): phan loai hay hoi quy, mac dinh la True (la classification)
            emotions (list): danh sach cam xuc can trich xuat, mac dinh la ['sad', 'neutral', 'happy']
            balance (bool): co nen can bang du lieu (ca dao tao va kiem tra), mac dinh la  True
        """

    def _load_data(self, desc_files, partition, shuffle):
        self.load_metadata_from_desc_file(desc_files, partition)
        # cân bằng cơ sở dữ liệu (cả đào tạo và kiểm tra)
        if partition == "train" and self.balance:
            self.balance_training_data()
        elif partition == "test" and self.balance:
            self.balance_testing_data()
        else:
            if self.balance:
                raise TypeError("Phan vung khong hop le, phai huan luyen, kiem tra")
        if shuffle:
            self.shuffle_data_by_partition(partition)

    # Tải dữ liệu huấn luyện từ tệp 'desc_files'
    def load_train_data(self, desc_files=["train_speech.csv"], shuffle=False):
        self._load_data(desc_files, "train", shuffle)
        
    def load_test_data(self, desc_files=["test_speech.csv"], shuffle=False):
    # Tải dữ liệu kiểm tra từ tệp `desc_files`
        self._load_data(desc_files, "test", shuffle)

    def shuffle_data_by_partition(self, partition):
        if partition == "train":
            self.train_audio_paths, self.train_emotions, self.train_features = shuffle_data(self.train_audio_paths,
            self.train_emotions, self.train_features)
        elif partition == "test":
            self.test_audio_paths, self.test_emotions, self.test_features = shuffle_data(self.test_audio_paths,
            self.test_emotions, self.test_features)
        else:
            raise TypeError("Phan vung khong hop le, phai huan luyen, kiem tra")

    def load_metadata_from_desc_file(self, desc_files, partition):
        """Doc du lieu tu tep CSV va trich xuat va tai cac tinh nang cua tep am thanh
        Thong so:
            desc_files (list): danh sach cac tep mo ta, (tep csv) de doc
            partition (str): train hoac test
        """
        # khung dữ liệu trống
        df = pd.DataFrame({'path': [], 'emotion': []})
        for desc_file in desc_files:
            # nối khung dữ liệu lại
            df = pd.concat((df, pd.read_csv(desc_file)), sort=False)
        if self.verbose:
            print("Tai duong dan du lieu am thanh va nhan tuong ung")
        # khởi tạo cột (columns)
        audio_paths, emotions = list(df['path']), list(df['emotion'])
        # không phải phân loại thì chuyển cảm xúc thành số
        if not self.classification:
            if len(self.emotions) == 3:
                self.categories = {'sad': 1, 'neutral': 2, 'happy': 3}
            else:
                raise TypeError("Hoi quy chi duoc dung cho ['sad', 'neutral', 'happy']")
            emotions = [ self.categories[e] for e in emotions ]
        # tạo thư mục nếu không có
        if not os.path.isdir(self.features_folder_name):
            os.mkdir(self.features_folder_name)
        # lấy nhãn cho các đặc điểm (feature)
        label = get_label(self.audio_config)
        # tạo tên tệp đặc điểm (feature)
        n_samples = len(audio_paths)
        first_letters = get_first_letters(self.emotions)
        name = os.path.join(self.features_folder_name, f"{partition}_{label}_{first_letters}_{n_samples}.npy")
        if os.path.isfile(name):
            # nếu tệp tồn tại, tải 
            if self.verbose:
                print("Tep da co, dang tai")
            features = np.load(name)
        else:
            # tệp không có thì trích xuất các đặc điểm và đưa vào tệp
            features = []
            append = features.append
            for audio_file in tqdm.tqdm(audio_paths, f"Trich xuat dac diem (feature) cho{partition}"):
                feature = extract_feature(audio_file, **self.audio_config)
                if self.input_dimension is None:
                    self.input_dimension = feature.shape[0]
                append(feature)
            # đổi sang mảng numpy
            features = np.array(features)
            # lưu
            np.save(name, features)
        if partition == "train":
            try:
                self.train_audio_paths
            except AttributeError:
                self.train_audio_paths = audio_paths
                self.train_emotions = emotions
                self.train_features = features
            else:
                if self.verbose:
                    print("Them mau dao tao bo sung")
                self.train_audio_paths += audio_paths
                self.train_emotions += emotions
                self.train_features = np.vstack((self.train_features, features))
        elif partition == "test":
            try:
                self.test_audio_paths
            except AttributeError:
                self.test_audio_paths = audio_paths
                self.test_emotions = emotions
                self.test_features = features
            else:
                if self.verbose:
                    print("Them mau kiem tra bo sung")
                self.test_audio_paths += audio_paths
                self.test_emotions += emotions
                self.test_features = np.vstack((self.test_features, features))
        else:
            raise TypeError("Phan vung khong hop le, phai co dao tao hoac kiem tra")

    def _balance_data(self, partition):
        if partition == "train":
            emotions = self.train_emotions
            features = self.train_features
            audio_paths = self.train_audio_paths
        elif partition == "test":
            emotions = self.test_emotions
            features = self.test_features
            audio_paths = self.test_audio_paths
        else:
            raise TypeError("Phan vung khong hop le, phai co dao tao hoac kiem tra")
        
        count = []
        if self.classification:
            for emotion in self.emotions:
                count.append(len([ e for e in emotions if e == emotion]))
        else:
            # sử dụng hồi quy, dùng số liệu thực, không đánh nhãn cảm xúc
            for emotion in self.categories.values():
                count.append(len([ e for e in emotions if e == emotion]))
        # lấy mẫu dữ liệu tối thiểu để cân bằng
        minimum = min(count)
        if minimum == 0:
            # nếu không cân bằng, 0 có mẫu (sample) được tải lên
            print("1 lop co 0 mau (sample), dat can bang thanh False")
            self.balance = False
            return
        if self.verbose:
            print("Can bang tap du lieu ve gia tri toi thieu", minimum)
        d = defaultdict(list)
        if self.classification:
            counter = {e: 0 for e in self.emotions }
        else:
            counter = { e: 0 for e in self.categories.values() }
        for emotion, feature, audio_path in zip(emotions, features, audio_paths):
            if counter[emotion] >= minimum:
                # vượt qua giá trị tối thiểu
                continue
            counter[emotion] += 1
            d[emotion].append((feature, audio_path))

        emotions, features, audio_paths = [], [], []
        for emotion, features_audio_paths in d.items():
            for feature, audio_path in features_audio_paths:
                emotions.append(emotion)
                features.append(feature)
                audio_paths.append(audio_path)
        
        if partition == "train":
            self.train_emotions = emotions
            self.train_features = features
            self.train_audio_paths = audio_paths
        elif partition == "test":
            self.test_emotions = emotions
            self.test_features = features
            self.test_audio_paths = audio_paths
        else:
            raise TypeError("Phan vung khong hop le, phai co dao tao hoac kiem tra")

    def balance_training_data(self):
        self._balance_data("train")

    def balance_testing_data(self):
        self._balance_data("test")
        

def shuffle_data(audio_paths, emotions, features):
    """ Tron du lieu duoc dung khi hoan thanh chuyen qua du lieu dao tao, xac nhan trong qua trinh dao tao
    Thong so:
        audio_paths (list): Duong dan den am thanh
        emotions (list): Cam xuc trong moi am thanh
        features (list): dac diem am thanh
    """
    p = np.random.permutation(len(audio_paths))
    audio_paths = [audio_paths[i] for i in p] 
    emotions = [emotions[i] for i in p]
    features = [features[i] for i in p]
    return audio_paths, emotions, features


def load_data(train_desc_files, test_desc_files, audio_config=None, classification=True, shuffle=True, balance=True, emotions=['sad', 'neutral', 'happy']):
    # tạo lớp cảm xúc
    audiogen = AudioExtractor(audio_config=audio_config, classification=classification, emotions=emotions, balance=balance, verbose=0)
    # tải dữ liệu đào tạo
    audiogen.load_train_data(train_desc_files, shuffle=shuffle)
    # tải dữ liệu kiểm tra
    audiogen.load_test_data(test_desc_files, shuffle=shuffle)
    # đưa ra X_train, X_test, y_train, y_test
    return {
        "X_train": np.array(audiogen.train_features),
        "X_test": np.array(audiogen.test_features),
        "y_train": np.array(audiogen.train_emotions),
        "y_test": np.array(audiogen.test_emotions),
        "train_audio_paths": audiogen.train_audio_paths,
        "test_audio_paths": audiogen.test_audio_paths,
        "balance": audiogen.balance,
    }