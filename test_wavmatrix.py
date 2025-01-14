from emotion_recognition import EmotionRecognizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

mt = EmotionRecognizer(emotions=["happy", "neutral", "sad"], balance=True, verbose=1, custom_db=True)

mt.train()

print(mt.confusion_matrix())
mt.draw_confusion_matrix()

prediction = mt.predict('ACuoi.wav')
print(f"Dự đoán: {prediction}")