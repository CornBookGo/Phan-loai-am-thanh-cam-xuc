import glob
import pandas as pd
import os


def write_emodb_csv(emotions=["sad", "neutral", "happy"], train_name="train_emo.csv", test_name="test_emo.csv", train_size=0.8, verbose=1):
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
    for file in glob.glob("data/emodb/*.wav"):
        try:
            emotion = categories[os.path.basename(file)[5]]
        except KeyError:
            continue
        target['emotion'].append(emotion)
        target['path'].append(file)
    if verbose:
        print("[Folder Emodb] Tổng số tập tin đã nhận:", len(target['path']))
        
    # chia tập huấn luyện, kiểm tra
    n_samples = len(target['path'])
    test_size = int((1-train_size) * n_samples)
    train_size = int(train_size * n_samples)
    if verbose:
        print("[Folder Emodb] Mẫu đào tạo:", train_size)
        print("[Folder Emodb] Mẫu kiểm tra:", test_size)   
    X_train = target['path'][:train_size]
    X_test = target['path'][train_size:]
    y_train = target['emotion'][:train_size]
    y_test = target['emotion'][train_size:]
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)


def write_tess_ravdess_csv(emotions=["sad", "neutral", "happy"], train_name="train_tess_ravdess.csv", test_name="test_tess_ravdess.csv", verbose=1):
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    
    for category in emotions:
        # thư mục âm thanh đào tạo
        total_files = glob.glob(f"data/training/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            train_target["path"].append(path)
            train_target["emotion"].append(category)
        if verbose and total_files:
            print(f" Có {len(total_files)} tệp âm thanh đào tạo cho nhãn:{category}")
    
        # thư mục âm thanh xác thực (validation)
        total_files = glob.glob(f"data/validation/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            test_target["path"].append(path)
            test_target["emotion"].append(category)
        if verbose and total_files:
            print(f" Có {len(total_files)} tệp âm thanh kiểm tra cho nhãn:{category}")
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)


def write_custom_csv(emotions=['sad', 'neutral', 'happy'], train_name="train_custom.csv", test_name="test_custom.csv", verbose=1):
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    for category in emotions:
        # đào tạo dữ liệu
        for i, file in enumerate(glob.glob(f"data/train-custom/*_{category}.wav")):
            train_target["path"].append(file)
            train_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[DL_Custom] Có {i} tệp âm thanh đào tạo cho nhãn:{category}")
            except NameError:
                # trường hợp {i} không tồn tại
                pass
        
        # kiểm tra dữ liệu
        for i, file in enumerate(glob.glob(f"data/test-custom/*_{category}.wav")):
            test_target["path"].append(file)
            test_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[DL_Custom] Có {i} tệp âm thanh kiểm tra cho nhãn:{category}")
            except NameError:
                pass
    
    # khởi tạo CSV
    if train_target["path"]:
        pd.DataFrame(train_target).to_csv(train_name)

    if test_target["path"]:
        pd.DataFrame(test_target).to_csv(test_name)