import pandas as pd

class OceanographicData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.depths = None
        self.instruments = None
        self.load_data()

        
    def load_data(self):
        # Загружаем данные из файла с указанием правильных названий столбцов
        self.data = pd.read_csv(
            self.file_path, 
            sep='\t', 
            parse_dates=['Date/Time'],
            usecols=['Date/Time', 'Depth water [m]', 'Gear ID', 'Press [dbar]', 'Temp [°C]', 'Sal'],
            skiprows=21
        )
        
        # Переименовываем столбцы для удобства работы
        self.data.columns = ['date_time', 'depth', 'instrument_index', 'pressure', 'temperature', 'salinity']
        
        # Преобразуем глубины в категории (поскольку они номинальные)
        self.data['depth'] = self.data['depth'].astype('category')
        
        # Получаем уникальные глубины и приборы
        self.depths = self.data['depth'].cat.categories
        self.instruments = self.data['instrument_index'].unique()
        
    def get_data_by_depth(self, depth):
        """Возвращает все данные для указанной глубины"""
        return self.data[self.data['depth'] == depth]
    
    def get_data_by_instrument(self, instrument_index):
        """Возвращает все данные для указанного прибора"""
        return self.data[self.data['instrument_index'] == instrument_index]
    
    def get_parameter(self, parameters, instrument_index=None, depth=None):
        """
        Возвращает данные для указанных параметров с возможной фильтрацией по прибору и глубине
        
        Параметры:
        parameters - строка или список строк: 'temperature', 'pressure', 'salinity'
        instrument_index - номер прибора (опционально)
        depth - глубина (опционально)
        """
        # Преобразуем одиночный параметр в список для унификации обработки
        if isinstance(parameters, str):
            parameters = [parameters]
            
        # Проверяем, что все запрошенные параметры существуют
        valid_params = {'temperature', 'pressure', 'salinity'}
        for param in parameters:
            if param not in valid_params:
                raise ValueError(f"Недопустимый параметр: {param}. Допустимые параметры: {valid_params}")
        
        # Создаем маску для фильтрации
        mask = ~self.data[parameters[0]].isna()
        for param in parameters[1:]:
            mask &= ~self.data[param].isna()
            
        # Обрабатываем фильтр по приборам
        if instrument_index is not None:
            if isinstance(instrument_index, (int, float)):
                instrument_index = [int(instrument_index)]
            mask &= self.data['instrument_index'].isin(instrument_index)

        if depth is not None:
            mask &= (self.data['depth'] == depth)
        
        # Выбираем только нужные столбцы
        columns = ['date_time', 'depth', 'instrument_index'] + parameters
        return self.data.loc[mask, columns].copy()
    
    def get_instruments_measuring(self, parameter):
        """Возвращает индексы приборов, которые измеряют указанный параметр"""
        instruments = {
            'temperature': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'salinity': [1, 2, 5, 7],
            'pressure': [2, 5, 7, 8, 9]
        }
        return instruments.get(parameter, [])
    
    def resample_data(self, freq='5T'):
        """
        Ресемплирует данные к указанной частоте (по умолчанию 5 минут).
        Для агрегации использует среднее значение.
        """
        resampled_dfs = []
        
        for depth in self.depths:
            for instrument in self.instruments:
                # Фильтруем данные по глубине и прибору
                mask = (self.data['depth'] == depth) & (self.data['instrument_index'] == instrument)
                instrument_data = self.data[mask].copy()
                
                if len(instrument_data) == 0:
                    continue
                
                # Ресемплируем с учетом исходной дискретности прибора
                resampled = instrument_data.set_index('date_time').resample(freq).mean()
                
                # Добавляем обратно информацию о глубине и приборе
                resampled['depth'] = depth
                resampled['instrument_index'] = instrument
                
                resampled_dfs.append(resampled.reset_index())
        
        # Объединяем все ресемплированные данные
        self.data = pd.concat(resampled_dfs, ignore_index=True)
        return self.data

    def get_original_column_names(self):
        """Возвращает соответствие между удобными именами и оригинальными названиями столбцов"""
        return {
            'date_time': 'Date/Time',
            'depth': 'Depth water [m]',
            'instrument_index': 'Gear ID',
            'pressure': 'Press [dbar]',
            'temperature': 'Temp [°C]',
            'salinity': 'Sal'
        }
    


