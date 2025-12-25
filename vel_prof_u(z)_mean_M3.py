import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import welch

dir = "data/M3_UV.tab"

data = pd.read_csv(dir, delimiter="\t", skiprows = 20)
print(data.keys())

def rotate(u_arr, v_arr, angl):
    x_arr = u_arr * np.cos(np.deg2rad(angl)) - v_arr * np.sin(np.deg2rad(angl))
    y_arr = v_arr * np.cos(np.deg2rad(angl)) + u_arr * np.sin(np.deg2rad(angl))
    return x_arr, y_arr

colors = [
    '#2E5A88',  # Январь - темно-синий (зима)
    '#4A77B4',  # Февраль - синий
    '#5AA6B8',  # Март - голубой (начало весны)
    '#6BC4A6',  # Апрель - бирюзовый
    '#8BD48E',  # Май - светло-зеленый (весна)
    '#B6E354',  # Июнь - салатовый (начало лета)
    '#FFE11A',  # Июль - желтый (лето)
    '#FFB142',  # Август - оранжевый
    '#FF7F50',  # Сентябрь - коралловый (осень)
    '#D9534F',  # Октябрь - красный
    '#A64D79',  # Ноябрь - пурпурный
    '#6A4D8F',  # Декабрь - фиолетовый (зима)
]

month_names = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

for id in range(1, 4):
    data1 = data[data["Gear ID"] == id]
    data1["Datetime"] = pd.to_datetime(data1["Date/Time"])
    data1["Date/Time"] = data1["Date/Time"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M").timestamp())
    datetime = data1["Datetime"].unique()
    

    data1 = data1.groupby(['Date/Time', 'Depth water [m]']).mean().reset_index()
    pivot_table_U = data1.pivot(index='Date/Time', columns='Depth water [m]', values='Cur vel U [cm/s]')
    pivot_table_V = data1.pivot(index='Date/Time', columns='Depth water [m]', values='Cur vel V [cm/s]')

    U = pivot_table_U.to_numpy().T
    V = pivot_table_V.to_numpy().T

    vel_along, vel_cross = rotate(U, V, 30)

    time_array = pivot_table_U.index.to_numpy()
    depth_array = pivot_table_U.columns.to_numpy()
    for m in range(2, 13):
        idx = np.array((datetime.month == m) & (datetime.year == 2009))
        mean_vel_along = np.nanmean(vel_along[:, idx], axis = 1)
        mean_vel_cross = np.nanmean(vel_cross[:, idx], axis = 1)
        plt.plot(mean_vel_along, depth_array, label = month_names[m], color = colors[m-1])
        plt.plot(mean_vel_cross, depth_array, '--', label = month_names[m], color = colors[m-1])


plt.xlabel("Velocity, cm/s")
plt.ylabel("Depth, m")

handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())

plt.ylim([730, 0])  
plt.show()