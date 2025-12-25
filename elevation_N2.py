import pandas as pd
import numpy as np
from read_hyd_data import *
import pandas as pd
import seawater as sw 
from scipy import stats

class SeawaterDensityCalculator:
    def __init__(self, ocean_data):
        self.data = ocean_data
    
    def calculate_density(self, temp, sal, depth):
        """Рассчитывает плотность морской воды с использованием библиотеки seawater"""
        # seawater (EOS-80 стандарт)
        density = sw.pden(sal, temp, depth)  # kg/m³
        return density
    
    def calibrate_density_model(self):
        """Калибрует линейную модель ρ = a*T + b по приборам 1,2,5"""
        # Получаем данные с приборов 1,2,5
        cal_data = self.data.get_parameter(
            ['temperature', 'salinity'],
            instrument_index=[1, 2, 5]
        ).dropna()
        
        # Рассчитываем истинную плотность
        cal_data['density'] = self.calculate_density(
            cal_data['temperature'],
            cal_data['salinity'],
            cal_data['depth']
        )
        
        # Линейная регрессия
        slope, intercept, r_value, _, _ = stats.linregress(
            cal_data['temperature'],
            cal_data['density']
        )
        
        return slope, intercept, r_value**2, cal_data
    
    def estimate_density(self, instrument_indices):
        """Оценивает плотность для указанных приборов (3,4) через температурную регрессию"""
        slope, intercept, r2, cal_data = self.calibrate_density_model()
        print(f"Регрессионная модель: ρ = {slope:.4f}·T + {intercept:.4f} (R²={r2:.3f})")
        
        # Получаем данные температуры для целевых приборов
        target_data = self.data.get_parameter(
            'temperature',
            instrument_index=instrument_indices
        ).dropna()
        
        # Оцениваем плотность
        target_data['estimated_density'] = slope * target_data['temperature'] + intercept
        
        return target_data, cal_data

class TemperatureGradientAnalyzer:
    def __init__(self, ocean_data):
        self.data = ocean_data
    
    def calculate_temperature_gradient(self):
        """Вычисляет вертикальный градиент температуры для приборов 1-5"""
        # Получаем данные для нужных приборов
        temp_data = self.data.get_parameter(
            'temperature',
            instrument_index=[1, 2, 3, 4, 5]
        )
        
        # Сортируем по времени и глубине
        temp_data = temp_data.sort_values(['date_time', 'depth'])
        
        # Вычисляем градиент температуры по глубине для каждого момента времени
        gradients = temp_data.groupby('date_time').apply(
            lambda x: self._compute_gradient(x['depth'], x['temperature'])
        )
        
        return gradients
    
    def _compute_gradient(self, depths, temperatures):
        """Вычисляет линейный градиент температуры по глубине"""
        if len(depths) < 2:
            return np.nan
        slope, _, _, _, _ = stats.linregress(depths, temperatures)
        return slope
    
    def calculate_gradient_deviation(self, window='48H'):
        """Вычисляет отклонение градиента от скользящего среднего"""
        gradients = self.calculate_temperature_gradient()
        
        # Вычисляем скользящее среднее
        rolling_mean = gradients.rolling(window, min_periods=1).mean()
        
        # Вычисляем отклонение
        deviation = gradients / rolling_mean
        
        return deviation
    
    def analyze_gradient_depth_dependence(self):
        """Анализирует зависимость градиента температуры от глубины"""
        temp_data = self.data.get_parameter(
            'temperature',
            instrument_index=[1, 2, 3, 4, 5]
        )
        print(temp_data)
        # Группируем по времени и вычисляем зависимость градиента от глубины
        results = []
        for time, group in temp_data.groupby('date_time'):
            if len(group) < 2:
                continue
                
            # Линейная регрессия для градиента по глубине
            slope, intercept, r_value, _, _ = stats.linregress(
                group['depth'],
                group['temperature']
            )
            
            results.append({
                'date_time': time,
                'gradient': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'mean_depth': group['depth'].mean()
            })
        
        return pd.DataFrame(results)
    