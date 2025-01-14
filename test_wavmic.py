from emotion_recognition import EmotionRecognizer

import pyaudio
import os
import wave
from sys import byteorder
from array import array
from struct import pack
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from utils import get_best_estimators

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "Trả về 'True' nếu dưới ngưỡng 'silent'"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Âm lượng âm thanh đầu ra"
    MAXIMUM = 16385
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Cắt bớt âm thanh ở đầu và cuối"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Cắt bên trái
    snd_data = _trim(snd_data)

    # Cắt bên phải
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Thêm khoảng trống vào đầu và cuối 'snd_data' theo giây dạng (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """Ghi âm 1 từ hoặc nhiều từ từ micro và trả về dữ liệu dưới dạng mảng âm ngắn được đánh dấu.
    Chuẩn hóa (Normalize) âm thanh, cắt bớt khoảng trống ở đầu và cuối, các phần đệm có 0,5 giây của âm trống đảm bảo VLC có thể phát mà không cần cắt nhỏ."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # độ bền nhỏ, âm ngắn được đánh dấu
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Ghi âm từ micro và đưa ra dữ liệu kết quả từ 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}



if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    import argparse
    parser = argparse.ArgumentParser(description="""Kiem tra he thong phan loai cam xuc khi dung giong noi, hay xem xet thay doi mo hinh hoac thong so theo y muon. """)
    parser.add_argument("-e", "--emotions", help="""Cam xuc de nhan dien duoc cach nhau bang dau phay ',', cam xuc de nhan dien la "sad" , "neutral", "happy" , mac dinh la "sad,neutral,happy"
""", default="sad,neutral,happy")
    parser.add_argument("-m", "--model", help="""Mo hinh duoc dung, 3 mo hinh duoc su dung la: {}, mac dinh la "SVC" """.format(estimators_str),default="SVC")


    # Phân tích các đối số được thông qua
    args = parser.parse_args()

    features = ["mfcc", "chroma", "mel"]
    detector = EmotionRecognizer(estimator_dict[args.model], emotions=args.emotions.split(","), features=features, verbose=0)
    detector.train()
    print("Độ chính xác khi kiểm tra dữ liệu: {:.3f}%".format(detector.test_score()*100))
    print("Hãy nói vào mic")
    
    filename = "test.wav"
    record_to_file(filename)
    result = detector.predict(filename)
    print(result)
    