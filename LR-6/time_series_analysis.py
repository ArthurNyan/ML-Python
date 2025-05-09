import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import seaborn as sns

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Загрузка данных
def load_data(filename):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Функция для проверки стационарности
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('Результаты теста Дики-Фуллера:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Критические значения:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    return result[1] < 0.05

# Функция для экспоненциального сглаживания
def exponential_smoothing(series, alpha=0.7):
    model = SimpleExpSmoothing(series)
    model_fit = model.fit(smoothing_level=alpha)
    return model_fit.forecast(1).iloc[0]

# Функция для построения модели AR
def build_ar_model(series, order):
    model = AutoReg(series, lags=order)
    model_fit = model.fit()
    return model_fit.predict(len(series), len(series)).iloc[0]

# Функция для визуализации временного ряда
def plot_time_series(series, title):
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Функция для построения графика PACF
def plot_pacf(series):
    plt.figure(figsize=(12, 6))
    nlags = min(10, len(series) // 2 - 1)  # Уменьшаем количество лагов
    pacf_values = pacf(series, nlags=nlags)
    plt.stem(range(len(pacf_values)), pacf_values)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    plt.title('График частичной автокорреляции (PACF)')
    plt.xlabel('Лаг')
    plt.ylabel('PACF')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Загрузка данных
    df = load_data('data/book_orders.csv')
    
    # Установка частоты для временного ряда
    df = df.asfreq('D')  # Устанавливаем дневную частоту
    
    # Отложим последнее значение для тестирования
    test_value = df['qty'].iloc[-1]
    train_data = df['qty'].iloc[:-1]
    
    # Анализ временного ряда
    print("\n=== Анализ временного ряда ===")
    plot_time_series(train_data, 'Временной ряд заказов книг')
    
    # Проверка стационарности
    print("\n=== Проверка стационарности ===")
    is_stationary = check_stationarity(train_data)
    print(f"Ряд {'стационарный' if is_stationary else 'нестационарный'}")
    
    # Определение порядка модели AR
    print("\n=== Определение порядка модели AR ===")
    plot_pacf(train_data)
    
    # Прогнозирование методом экспоненциального сглаживания
    exp_smooth_pred = exponential_smoothing(train_data)
    print("\n=== Результаты прогнозирования ===")
    print(f"Экспоненциальное сглаживание (α=0.7): {exp_smooth_pred:.2f}")
    
    # Построение модели AR (используем порядок p=1 как пример)
    ar_pred = build_ar_model(train_data, order=1)
    print(f"Прогноз AR(1): {ar_pred:.2f}")
    print(f"Фактическое значение: {test_value:.2f}")
    
    # Сравнение ошибок
    exp_smooth_error = abs(test_value - exp_smooth_pred)
    ar_error = abs(test_value - ar_pred)
    print("\n=== Сравнение ошибок ===")
    print(f"Ошибка экспоненциального сглаживания: {exp_smooth_error:.2f}")
    print(f"Ошибка модели AR: {ar_error:.2f}") 