import glob
import pandas as pd
import os


def write_emodb_csv(emotions=["sad", "neutral", "happy"], train_name="train_emo.csv",
                    test_name="test_emo.csv", train_size=0.8, verbose=1):
    """
    Doc tap du lieu giong noi emodb tu thu muc va ghi vao tep CSV.
    Thong so:
        emotions (list): danh sach cam xuc de doc tu thu muc, mac dinh la ['sad', 'neutral', 'happy']
        train_name (str): ten tep CSV dau ra cho du lieu huan luyen, mac dinh la 'train_emo.csv'
        test_name (str): ten tep CSV dau ra cho du lieu thu nghiem, mac dinh la 'test_emo.csv'
        train_size (float): ty le chia du lieu cho huan luyen, mac dinh la 0.8 (80% Du lieu dao tao va 20% Du lieu kiem tra)
        verbose (int/bool): Muc do tu ngu, 0 cho im lang, 1 cho thong tin, mac dinh la 1
    """
    target = {"path": [], "emotion": []}
    categories = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral"
    }
    # xóa các cảm xúc không có ký tự trên
    categories_reversed = { v: k for k, v in categories.items() }
    for emotion, code in categories_reversed.items():
        if emotion not in emotions:
            del categories[code]
    for file in glob.glob("data/emodb/wav/*.wav"):
        try:
            emotion = categories[os.path.basename(file)[5]]
        except KeyError:
            continue
        target['emotion'].append(emotion)
        target['path'].append(file)
    if verbose:
        print("[Emodb] Tổng số tập tin đã nhận:", len(target['path']))
        
    # dividing training/testing sets
    n_samples = len(target['path'])
    test_size = int((1-train_size) * n_samples)
    train_size = int(train_size * n_samples)
    if verbose:
        print("[Emodb] Mẫu đào tạo:", train_size)
        print("[Emodb] Mẫu kiểm tra:", test_size)   
    X_train = target['path'][:train_size]
    X_test = target['path'][train_size:]
    y_train = target['emotion'][:train_size]
    y_test = target['emotion'][train_size:]
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)


def write_tess_ravdess_csv(emotions=["sad", "neutral", "happy"], train_name="train_tess_ravdess.csv", test_name="test_tess_ravdess.csv", verbose=1):
    """
    Doc bo du lieu giong noi cua Tess va Ravdess va ghi vao tep CSV
    Thong so:
        emotions (list): danh sach cam xuc doc tu thu muc, mac dinh la ['sad', 'neutral', 'happy']
        train_name (str): ten tep CSV dau ra cho du lieu huan luyen, mac dinh la 'train_tess_ravdess.csv'
        test_name (str): ten tep CSV dau ra cho du lieu kiem tra, mac dinh la 'test_tess_ravdess.csv'
        verbose (int/bool): Muc do tu ngu, 0 cho im lang, 1 cho thong tin, mac dinh la 1
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    
    for category in emotions:
        # thư mục âm thanh đào tạo
        total_files = glob.glob(f"data/training/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            train_target["path"].append(path)
            train_target["emotion"].append(category)
        if verbose and total_files:
            print(f" Co {len(total_files)} tep am thanh dao tao cho nhan:{category}")
    
        # thư mục âm thanh chấp nhận (validation)
        total_files = glob.glob(f"data/validation/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            test_target["path"].append(path)
            test_target["emotion"].append(category)
        if verbose and total_files:
            print(f" Co {len(total_files)} tep am thanh kiem tra cho nhan:{category}")
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)


def write_custom_csv(emotions=['sad', 'neutral', 'happy'], train_name="train_custom.csv", test_name="test_custom.csv",
                    verbose=1):
    """
    Doc du lieu am thanh tu tao tu data/*-custom va cac tap tin mo ta (csv)
    Thong so:
        emotions (list): danh sach cam xuc doc tu thu muc, mac dinh la ['sad', 'neutral', 'happy']
        train_name (str): ten tep CSV dau ra cho du lieu huan luyen, mac dinh la 'train_custom.csv'
        test_name (str): ten tep CSV dau ra cho du lieu kiem tra, mac dinh la  'test_custom.csv'
        verbose (int/bool): Muc do tu ngu, 0 cho im lang, 1 cho thong tin, mac dinh la 1
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    for category in emotions:
        # đào tạo dữ liệu
        for i, file in enumerate(glob.glob(f"data/train-custom/*_{category}.wav")):
            train_target["path"].append(file)
            train_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[Cust_Data] Co {i} tep am thanh dao tao cho nhan:{category}")
            except NameError:
                # trường hợp {i} không tồn tại
                pass
        
        # kiểm tra dữ liệu
        for i, file in enumerate(glob.glob(f"data/test-custom/*_{category}.wav")):
            test_target["path"].append(file)
            test_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[Cust_Data] Co {i} tep am thanh kiem tra cho nhan:{category}")
            except NameError:
                pass
    
    # khởi tạo CSV
    if train_target["path"]:
        pd.DataFrame(train_target).to_csv(train_name)

    if test_target["path"]:
        pd.DataFrame(test_target).to_csv(test_name)