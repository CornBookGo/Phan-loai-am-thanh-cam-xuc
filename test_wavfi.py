from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

cl_model = SVC()

fi = EmotionRecognizer(model=cl_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)

fi.train()

print("Dự đoán:", fi.predict("ACuoi.wav"))