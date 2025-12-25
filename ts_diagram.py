import read_hyd_data as rhd
import matplotlib.pyplot as plt
import seawater as sw
import numpy as np

dir = "dataM3_TSP.tab"
data_processor = rhd.OceanographicData(dir)

# data_processor

ts = data_processor.get_parameter(['temperature', 'salinity'])
ts['den'] = sw.pden(ts['salinity'], ts['temperature'], ts['depth'])

x = ts['temperature']
y = ts['den']

A = np.vstack([x, np.ones(len(x))]).T
k, b = np.linalg.lstsq(A, y, rcond=None)[0]

x_line = np.linspace(min(x), max(x), 100)
y_line = k * x_line + b
print(k, b)
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Линейная аппроксимация: y = {k:.4f}x + {b:.4f}')


coef = np.polyfit(x, y, 2)
a, b, c = coef
print(coef)
x_line = np.linspace(min(x), max(x), 100)
y_line = a * x_line**2 + b * x_line + c

plt.scatter(ts['temperature'], ts['den'], c =ts['depth'], s = .1)

plt.ylabel('Density, kg\m3')
plt.xlabel('Temperature, C')

plt.colorbar(label = 'Depth, m')
plt.show()