import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import datetime as dt
import seawater

dir = "data/M3_TSP.tab"
df = pd.read_table(dir, sep='\t', skiprows = 22, names = ["Datetime","Depth", "ID","Press","Temp",	"Sal"])

print(df)
df['Datetime'] = pd.to_datetime(df['Datetime'])

T_test = df[df["Depth"] == 715]['Temp']

T = np.empty([10, T_test.shape[0]])
dep = np.sort(df['Depth'].unique())

timeStart = dt.datetime(2009, 2, 13, 18, 0)
# Time = np.array(df[df["Depth"] == 715]['Datetime'], dtype='datetime')
Time = df[df["Depth"] == 715]['Datetime']
time_ref = (df[df["Depth"] == 715]['Datetime'] - timeStart).dt.total_seconds()

print(dep)

for i, d in enumerate(dep):
    if i in [0, 2, 4, 5, 6, 7, 9]:
        temp = seawater.eos80.ptmp(34.5, df[df["Depth"] == d]['Temp'], d) # 5 min
        T[i, :] = temp

for i, d in enumerate(dep):
    if i in [3]:
        temp = seawater.eos80.ptmp(34.5, df[df["Depth"] == d]['Temp'], d) # 20 min
        timethis = (df[df["Depth"] == d]['Datetime'] - timeStart).dt.total_seconds()
        T[i, :] = interpolate.interp1d(timethis, temp, fill_value=np.nan)(time_ref) 


for i, d in enumerate(dep):
    if i in [1, 8]:
        temp = seawater.eos80.ptmp(34.5, df[df["Depth"] == d]['Temp'], d) # 60 min
        timethis = (df[df["Depth"] == d]['Datetime'] - timeStart).dt.total_seconds()
        T[i, :] = interpolate.interp1d(timethis, temp, fill_value='extrapolate')(time_ref) 


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
f = plt.figure(figsize=(4, 4))
for m in range(2, 13):
    idx = np.array((Time.dt.month == m) & (Time.dt.year == 2009))
    temp_month_aver = np.nanmean(T[:, idx], axis = 1)
    plt.plot(temp_month_aver, dep, label = month_names[m], color = colors[m-1])
plt.plot(np.zeros_like(dep)+0.1, dep, 'ko')
plt.grid()
plt.ylim([720, 350])
plt.legend(title='2009', 
           fontsize=10,
           bbox_to_anchor=(1.05, 1),
           loc='upper left')
plt.ylabel('Depth, m')
plt.xlabel('Temperature, $^\circ$C')
plt.tight_layout()
# plt.savefig("mean_month_temp_profile.png", dpi = 800, transparent = 1)
plt.show()

