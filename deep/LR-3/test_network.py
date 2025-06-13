import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from neural_network import NeuralNetwork

# Генерация данных
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = X.T
y = y.reshape(1, -1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.T, y_test.T

# Создание и обучение сети
nn = NeuralNetwork(layer_dims=[2, 10, 5, 1], activation='relu', learning_rate=0.01)
history = nn.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=True)

# Оценка точности
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

train_accuracy = accuracy_score(y_train.flatten(), y_pred_train.flatten())
test_accuracy = accuracy_score(y_test.flatten(), y_pred_test.flatten())

print(f'\nТочность на обучающей выборке: {train_accuracy:.4f}')
print(f'Точность на тестовой выборке: {test_accuracy:.4f}')

# Визуализация границы решений
def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    # График границы решений
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[0], X[1], c=y.flatten(), cmap=plt.cm.RdYlBu)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Граница решений')
    
    # График функции потерь
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.xlabel('Эпоха')
    plt.ylabel('Функция потерь')
    plt.title('Процесс обучения')
    
    plt.tight_layout()
    plt.show()

# Визуализация результатов
plot_decision_boundary(X_test, y_test, nn) 