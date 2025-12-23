from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

iris = load_iris()
X = iris.data

Y = (iris.target == 0).astype(int)
Y = Y.reshape(Y.size, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
per = Perceptron(random_state = 42)
per.fit(X_train, Y_train)

Y_pred = per.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
