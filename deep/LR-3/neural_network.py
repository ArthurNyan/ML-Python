import numpy as np

class ActivationFunction:
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def sigmoid_derivative(Z):
        s = ActivationFunction.sigmoid(Z)
        return s * (1 - s)
    
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)
    
    @staticmethod
    def relu_derivative(Z):
        return np.where(Z > 0, 1, 0)
    
    @staticmethod
    def tanh(Z):
        return np.tanh(Z)
    
    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.tanh(Z)**2

class NeuralNetwork:
    def __init__(self, layer_dims, activation='relu', learning_rate=0.01):
        """
        Инициализация нейронной сети
        
        Parameters:
        layer_dims -- список размерностей слоев [n_x, n_h1, n_h2, ..., n_y]
        activation -- функция активации для скрытых слоев ('relu' или 'tanh')
        learning_rate -- скорость обучения
        """
        self.L = len(layer_dims) - 1
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = {}
        
        # Выбор функции активации
        if activation == 'relu':
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        
        # Инициализация параметров (веса и смещения)
        for l in range(1, self.L + 1):
            # He initialization для ReLU
            self.parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    def _forward(self, X):
        """
        Прямое распространение
        
        Parameters:
        X -- входные данные, shape (n_x, m)
        
        Returns:
        AL -- выходной слой
        cache -- кэш для обратного распространения
        """
        cache = {}
        A = X
        cache['A0'] = X
        
        # Скрытые слои с выбранной функцией активации
        for l in range(1, self.L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A) + b
            A = self.activation(Z)
            
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        # Выходной слой с сигмоидной функцией активации
        WL = self.parameters[f'W{self.L}']
        bL = self.parameters[f'b{self.L}']
        
        ZL = np.dot(WL, A) + bL
        AL = ActivationFunction.sigmoid(ZL)
        
        cache[f'Z{self.L}'] = ZL
        cache[f'A{self.L}'] = AL
        
        return AL, cache
    
    def _compute_cost(self, AL, Y):
        """
        Вычисление функции потерь (бинарная кросс-энтропия)
        
        Parameters:
        AL -- выход сети
        Y -- истинные метки
        
        Returns:
        cost -- значение функции потерь
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(AL + 1e-15) + (1 - Y) * np.log(1 - AL + 1e-15))
        return cost
    
    def _backward(self, AL, Y, cache):
        """
        Обратное распространение ошибки
        
        Parameters:
        AL -- выход сети
        Y -- истинные метки
        cache -- кэш из прямого распространения
        
        Returns:
        grads -- градиенты для обновления параметров
        """
        m = Y.shape[1]
        grads = {}
        
        # Градиент выходного слоя
        dAL = -(np.divide(Y, AL + 1e-15) - np.divide(1 - Y, 1 - AL + 1e-15))
        dZL = dAL * ActivationFunction.sigmoid_derivative(cache[f'Z{self.L}'])
        
        grads[f'dW{self.L}'] = 1/m * np.dot(dZL, cache[f'A{self.L-1}'].T)
        grads[f'db{self.L}'] = 1/m * np.sum(dZL, axis=1, keepdims=True)
        
        # Градиенты скрытых слоев
        for l in reversed(range(1, self.L)):
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZL)
            dZ = dA * self.activation_derivative(cache[f'Z{l}'])
            
            grads[f'dW{l}'] = 1/m * np.dot(dZ, cache[f'A{l-1}'].T)
            grads[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            
            dZL = dZ
        
        return grads
    
    def _update_parameters(self, grads):
        """
        Обновление параметров с помощью градиентного спуска
        
        Parameters:
        grads -- градиенты
        """
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
    
    def fit(self, X_train, Y_train, epochs=1000, batch_size=32, verbose=True):
        """
        Обучение нейронной сети
        
        Parameters:
        X_train -- обучающие данные
        Y_train -- метки
        epochs -- количество эпох
        batch_size -- размер мини-батча
        verbose -- выводить ли прогресс
        
        Returns:
        history -- история обучения (значения функции потерь)
        """
        m = X_train.shape[1]
        history = []
        
        for epoch in range(epochs):
            # Перемешивание данных
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]
            
            # Разбиение на мини-батчи
            num_batches = m // batch_size
            
            epoch_cost = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                Y_batch = Y_shuffled[:, start_idx:end_idx]
                
                # Прямое распространение
                AL, cache = self._forward(X_batch)
                
                # Вычисление функции потерь
                cost = self._compute_cost(AL, Y_batch)
                epoch_cost += cost
                
                # Обратное распространение
                grads = self._backward(AL, Y_batch, cache)
                
                # Обновление параметров
                self._update_parameters(grads)
            
            epoch_cost /= num_batches
            history.append(epoch_cost)
            
            if verbose and epoch % 100 == 0:
                print(f'Эпоха {epoch}, функция потерь: {epoch_cost:.4f}')
        
        return history
    
    def predict(self, X):
        """
        Предсказание для новых данных
        
        Parameters:
        X -- входные данные
        
        Returns:
        predictions -- предсказанные метки (0 или 1)
        """
        AL, _ = self._forward(X)
        predictions = (AL > 0.5).astype(int)
        return predictions 