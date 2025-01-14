# Nhận diện âm thanh cảm xúc
## Giới thiệu sản phẩm
- Ứng dụng này xử lý việc xây dựng và đào tạo Hệ thống nhận dạng cảm xúc lời nói.
- Ý tưởng cơ bản đằng sau công cụ này là xây dựng và đào tạo/thử nghiệm thuật toán học máy (cũng như học sâu) phù hợp có thể nhận dạng và phát hiện cảm xúc của con người từ lời nói.
- Điều này rất hữu ích cho nhiều lĩnh vực công nghiệp như đưa ra đề xuất sản phẩm, tính toán tình cảm, v.v.

## Yêu cầu
- **Python 3.8**
### Gói thư viện trong Python
- **tensorflow==2.5.2**
- **librosa==0.6.3**
- **numpy==1.20.0**
- **pandas==2.0.3**
- **soundfile==0.9.0**
- **wave**
- **scikit-learn==0.24.2**
- **tqdm==4.28.1**
- **matplotlib==2.2.3**
- **pyaudio==0.2.11**
- **[ffmpeg](https://ffmpeg.org/): được sử dụng nếu muốn thêm nhiều âm thanh mẫu hơn bằng cách chuyển đổi sang tốc độ mẫu 16000Hz và kênh đơn âm (mono) được dùng trong``convert_wavs.py``

Cài đặt các thư viện này bằng lệnh sau:
```
pip install -r requirements.txt
```

### Tập dữ liệu
Tập lưu trữ đã sử dụng 4 bộ dữ liệu (bao gồm cả tập dữ liệu tùy chỉnh của kho lưu trữ này) đã được tải xuống và định dạng sẵn trong thư mục `data`:
- [**RAVDESS**](https://zenodo.org/record/1188976) : Tập dữ liệu âm thanh cảm xúc có 24 diễn viên (12 nam, 12 nữ), phát âm hai câu nói khớp từ vựng với giọng Bắc Mỹ trung tính.
- [**TESS**](https://tspace.library.utoronto.ca/handle/1807/24487) : Dữ liệu âm thanh cảm xúc Toronto được mô phỏng theo Bài kiểm tra thính giác số 6 của Đại học Northwestern (NU-6; Tillman & Carhart, 1966). Một bộ gồm 200 từ mục tiêu được nói trong cụm từ mang "Nói từ _____" của hai nữ diễn viên (26 và 64 tuổi).
- [**EMO-DB**](http://emodb.bilderbar.info/docu/) : Là một phần của dự án nghiên cứu SE462/3-1 do DFG tài trợ vào năm 1997 và 1999, dữ liệu ghi lại cơ sở dữ liệu về những lời phát biểu cảm xúc của các diễn viên. Quá trình ghi âm diễn ra trong phòng cách âm của Đại học Kỹ thuật Berlin, khoa Âm học Kỹ thuật.
- **Custom** : Một số tập dữ liệu nhiễu không cân bằng nằm trong `data/train-custom` để huấn luyện và `data/test-custom` để kiểm tra, trong đó bạn có thể thêm/xóa các mẫu ghi âm một cách dễ dàng bằng cách chuyển đổi âm thanh thô sang tốc độ mẫu 16000, kênh đơn âm (được cung cấp trong code `create_wavs.py` trong phương thức ``convert_audio(audio_path)``, yêu cầu bắt buộc phải cài đặt [ffmpeg](https://ffmpeg.org/) và đặt bộ xử lý trong *PATH*) và thêm cảm xúc vào cuối tên tệp âm thanh được phân tách bằng '_' (ví dụ: "12345678_123456_happy.wav" sẽ được phân tích cú pháp tự động thành hạnh phúc)


### Cảm xúc để nhận diện
Có 9 cảm xúc có sẵn: "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (ngạc nhiên) and "boredom".
## Trích xuất tính năng
Trích xuất đặc trưng là phần chính của hệ thống nhận dạng cảm xúc lời nói. Về cơ bản, nó được thực hiện bằng cách thay đổi dạng sóng giọng nói thành dạng biểu diễn tham số ở tốc độ dữ liệu tương đối thấp hơn.
Trong kho lưu trữ này đã sử dụng các tính năng được sử dụng nhiều nhất hiện có trong thư viện [librosa](https://github.com/librosa/librosa) bao gồm:
- [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Chromagram 
- MEL Spectrogram Frequency (mel)
- Contrast
- Tonnetz (tonal centroid features)

## Tìm kiếm lưới
Kết quả tìm kiếm lưới đã được cung cấp trong thư mục `grid`, nhưng nếu muốn điều chỉnh các tham số tìm kiếm lưới khác nhau trong `parameters.py`, có thể chạy tập lệnh `grid_search.py` bằng cách:
```
python grid_search.py
```
Việc này có thể mất vài giờ để hoàn tất quá trình thực thi, sau khi hoàn tất, các công cụ ước tính tốt nhất sẽ được lưu trữ và chọn trong thư mục `grid` của 2 bộ phân loại `best_classification.pickle` và hồi quy `best_regressors.pickle`

## Ví dụ : Sử dụng 3 Cảm xúc
Cách xây dựng và huấn luyện mô hình phân loại 3 cảm xúc như sau:

```python
from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
# khởi tạo một mô hình, sử dụng SVC
my_model = SVC()
# chuyển mô hình tới EmotionRecognizer và cân bằng tập dữ liệu
rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
# đào tạo mô hình
rec.train()
# kiểm tra độ chính xác kiểm tra cho mô hình đó
print("Giá trị kiểm tra:", rec.test_score())
# kiểm tra độ chính xác của đào tạo cho mô hình đó
print("Giá trị đào tạo:", rec.train_score())
```
**Đầu ra:**
```
Giá trị kiểm tra: 0.8148148148148148
Giá trị đào tạo: 1.0
```
### Xác định mô hình tốt nhất
Để xác định mô hình tốt nhất, có thể dùng câu lệnh bằng cách:

```python
# tải các công cụ ước tính tốt nhất từ ​​thư mục `grid` đã được GridSearchCV tìm kiếm trong `grid_search.py`,
# và đặt mô hình ở mức tốt nhất về điểm kiểm tra, sau đó huấn luyện nó
rec.determine_best_model()
# lấy tên mô hình sklearn đã xác định
print(rec.model.__class__.__name__, "là tốt nhất")
# lấy điểm chính xác của bài kiểm tra có độ ước tính tốt nhất
print("Giá trị kiểm tra:", rec.test_score())

### Dự đoán
Chỉ cần truyền một đường dẫn âm thanh đến phương thức `rec.predict()` như bên dưới:
```python
# đây là bài phát biểu trung lập từ emo-db từ bộ thử nghiệm
print("Dự đoán:", rec.predict("data/emodb/wav/15a04Nc.wav"))
# đây là một tệp âm thanh buồn của TESS từ bộ thử nghiệm
print("Dự đoán:", rec.predict("data/validation/Actor_25/25_01_01_01_back_sad.wav"))
```
**Đầu ra:**
```
Dự đoán: neutral
Dự đoán: sad
```
Bạn có thể chuyển bất kỳ tệp âm thanh nào, nếu nó không ở định dạng thích hợp (16000Hz và kênh đơn âm), thì nó sẽ tự động được chuyển đổi, đảm bảo bạn đã cài đặt `ffmpeg` trong hệ thống của mình và được thêm vào *PATH*.
## Ví dụ 3: Không chuyển bất kỳ mô hình nào và xóa tập dữ liệu tùy chỉnh
Mã bên dưới khởi tạo `EmotionRecognizer` với 3 cảm xúc đã chọn trong khi xóa Tập dữ liệu tùy chỉnh và đặt `balance` thành `False`:
```python
from emotion_recognition import EmotionRecognizer
# khởi tạo phiên bản, việc này sẽ mất một chút thời gian trong lần thực thi đầu tiên
#vì nó sẽ tự động trích xuất các tính năng và gọi hàm xác định_best_model()
rec = EmotionRecognizer(emotions=["angry", "neutral", "sad"], balance=False, verbose=1, custom_db=False)
# nó sẽ được đào tạo, vì vậy không cần phải đào tạo lần này
# đưa ra được độ chính xác trên tập kiểm tra
print(rec.confusion_matrix())
# dự đoán mẫu âm thanh tức giận
prediction = rec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
print(f"Dự đoán: {prediction}")
```
**Đầu ra:**
```
[+] Mô hình tốt nhất được xác định: RandomForestClassifier với độ chính xác kiểm tra 93,454%

              predicted_angry  predicted_neutral  predicted_sad
true_angry          98.275864           1.149425       0.574713
true_neutral         0.917431          88.073395      11.009174
true_sad             6.250000           1.875000      91.875000

Prediction: angry
```
Bạn có thể in số lượng mẫu trên mỗi lớp:
```python
rec.get_samples_by_class()
```
**Output:**
```
         train  test  total
angry      910   174   1084
neutral    650   109    759
sad        862   160   1022
total     2422   443   2865
```
Trong trường hợp này, tập dữ liệu chỉ từ TESS và RAVDESS và không được cân bằng, bạn có thể chuyển `True` sang `balance` trên phiên bản `EmotionRecognizer` để cân bằng dữ liệu.
## Thuật toán được sử dụng
Kho lưu trữ này có thể được sử dụng để xây dựng các bộ phân loại machine learning cũng như các bộ hồi quy cho trường hợp 3 cảm xúc  {'sad': 0, 'neutral': 1, 'happy': 2}
### Mô hình phân loại
- SVC
- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- MLPClassifier
- BaggingClassifier
- Recurrent Neural Networks (Keras)
### Biến hồi quy
- SVR
- RandomForestRegressor
- GradientBoostingRegressor
- KNeighborsRegressor
- MLPRegressor
- BaggingRegressor
- Recurrent Neural Networks (Keras)

### Kiểm tra
Có thể kiểm tra giọng nói của chính mình bằng cách thực hiện lệnh sau:
```
python test.py
```
Đợi cho đến khi lời nhắc "Xin vui lòng nói chuyện" xuất hiện, sau đó bạn có thể bắt đầu nói và mô hình sẽ tự động phát hiện cảm xúc của bạn khi bạn dừng (nói).

Bạn có thể thay đổi cảm xúc để dự đoán, cũng như mô hình, gõ ``--help`` để biết thêm thông tin.
```
python test.py --help
```
**Đầu ra:**
```
usage: test.py [-h] [-e EMOTIONS] [-m MODEL]

Testing emotion recognition system using your voice, please consider changing
the model and/or parameters as you wish.

optional arguments:
  -h, --help            show this help message and exit
  -e EMOTIONS, --emotions EMOTIONS
                        Emotions to recognize separated by a comma ',',
                        available emotions are "neutral", "calm", "happy"
                        "sad", "angry", "fear", "disgust", "ps" (pleasant
                        surprise) and "boredom", default is
                        "sad,neutral,happy"
  -m MODEL, --model MODEL
                        The model to use, 8 models available are: "SVC","AdaBo
                        ostClassifier","RandomForestClassifier","GradientBoost
                        ingClassifier","DecisionTreeClassifier","KNeighborsCla
                        ssifier","MLPClassifier","BaggingClassifier", default
                        is "BaggingClassifier"

```

## Vẽ biểu đồ
Điều này sẽ chỉ hoạt động nếu "Grid_search" được thực hiện.
```python
from emotion_recognition import plot_histograms
# vẽ biểu đồ trên các phân loại khác nhau
plot_histograms(classifiers=True)
```
**Đầu ra:**
Biểu đồ hiển thị kết quả số liệu của các thuật toán khác nhau trên các kích thước dữ liệu khác nhau cũng như thời gian sử dụng để đào tạo/dự đoán.</p>

