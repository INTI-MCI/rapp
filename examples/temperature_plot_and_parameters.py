import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

''' 
This example provides correction parameters for the temperature sensor used 
in polarimeter measurements (DS18B20 sensor).
The data used consists of the temperature measured by the DS18B20 sensor (filepath1) 
and by two calibrated temperature sensors (filepath2) during a certain period of time.
The temperature measurements are sampled simultaneously, but at different times, 
so the signals were interpolated using a single time axis for further analysis.
We chose to analyze the bias correction and a linear correction for the DS18B20 sensor
and found that the best results were obtained with a linear correction. 
'''

filepath1 = r'C:\Users\Admin\rapp\workdir\output-data\2024-11-14-1731605585.2825165-temperatura-tiempo-total-9000-tiempo-espera-120-muestras10.txt'
filepath2 = r'C:\Users\Admin\rapp\workdir\output-data\2024-11-14-mediciones-temp-sensores-calibrados.txt'


def time_to_seconds(time_str) -> int:
    h, m, s = map(int, time_str.split(':'))
    return h*3600 + m*60 + s

'''Import ds18b20 data and calculate average and std:'''
data1_ = np.genfromtxt(filepath1, delimiter=',', skip_header=1, dtype=str)
data1 = np.array(data1_.tolist())
data_DS18B20 = data1[:, :10].astype(float)
N = data1.shape[0]

# time1 = np.loadtxt(filepath1, dtype=str, usecols=-1, skiprows=1, delimiter=',')
timestamps_ds18b20 = data1[:, 10]
time_ds18b20 = np.array([datetime.strptime(t, '%H:%M:%S') for t in timestamps_ds18b20])

average_ds18b20 = np.mean(data_DS18B20, axis=1)
av_ds18b20_wo_mean = average_ds18b20 - np.mean(average_ds18b20)
stds_ds18b20 = np.std(data_DS18B20, axis=1)
pooled_std = np.sqrt(np.sum(stds_ds18b20**2) / N)
print('Desviacion combinada =', pooled_std)
print('Promedio de las dispersiones =', np.mean(stds_ds18b20))

'''Import calibrated sensors data:'''
data2 = np.genfromtxt(filepath2, delimiter=',')
# skip_header=17, skip_footer=16) (para las mediciones del 7-11)
sensor1 = data2[:, 0]
sensor1_wo_mean = sensor1 - np.mean(data2[:, 0])
sensor2 = data2[:, 1]
sensor2_wo_mean = sensor2 - np.mean(data2[:, 1])

time2 = np.loadtxt(filepath2, dtype=str, usecols=3, delimiter=',')
time_calibrated_sensors = np.array([datetime.strptime(t, '%H:%M:%S') for t in time2])

plt.errorbar(time_ds18b20, av_ds18b20_wo_mean, yerr=stds_ds18b20/np.sqrt(10), label='Promedio DS18B20')
plt.plot(time_calibrated_sensors, sensor1_wo_mean, label='Sensor calibrado 1')
plt.plot(time_calibrated_sensors, sensor2_wo_mean, label='Sensor calibrado 2')
plt.title("Mediciones temperatura (ºC) vs tiempo")
plt.legend()
plt.show()


'''Interpolated signals:'''
time_seconds_ds18b20 = np.array([time_to_seconds(t) for t in timestamps_ds18b20])
time_seconds_calib_s = np.array([time_to_seconds(t) for t in time2])

min_time_range = max(time_seconds_ds18b20[0], time_seconds_calib_s[0])
max_time_range = min(time_seconds_ds18b20[-1], time_seconds_calib_s[-1])
time_interp = np.linspace(min_time_range, max_time_range, N)

splined_ds18b20 = np.interp(time_interp, time_seconds_ds18b20, average_ds18b20)
splined_calib1 = np.interp(time_interp, time_seconds_calib_s, sensor1)
splined_calib2 = np.interp(time_interp, time_seconds_calib_s, sensor2)

plt.plot(time_interp, splined_ds18b20, label='Promedio DS18B20')
plt.plot(time_interp, splined_calib1, label='Sensor calibrado 1')
plt.plot(time_interp, splined_calib2, label='Sensor calibrado 2')
plt.title("Temperatura interpolada (ºC) vs tiempo (s)")
plt.legend()
plt.show()


'''Bias correction for DS18B20 measurements using the interpolated signals:'''
bias_1 = splined_ds18b20 - splined_calib1
bias_2 = splined_ds18b20 - splined_calib2
bias = np.mean((np.mean(bias_1), np.mean(bias_2)))

ds18b20_wo_bias = splined_ds18b20 - bias

''' Linear correction for DS18B20 measurements using the interpolated signals:'''
A_ds18b20 = np.hstack([splined_ds18b20[:, np.newaxis], np.ones((N, 1))])

A1 = np.linalg.inv(A_ds18b20.T @ A_ds18b20) @ A_ds18b20.T @ splined_calib1[:, np.newaxis]
A2 = np.linalg.inv(A_ds18b20.T @ A_ds18b20) @ A_ds18b20.T @ splined_calib2[:, np.newaxis]

linear_correction = (A1 + A2) / 2
print('[[A] [b]] =', linear_correction)

ds18b20_transformed1 = A_ds18b20 @ A1
ds18b20_transformed2 = A_ds18b20 @ A2

ds18b20_linear_correction = A_ds18b20 @ linear_correction

error = ds18b20_linear_correction - splined_calib1
error = np.linalg.norm(error) / np.sqrt(N)
print('RMSE =', error)

plt.plot(time_interp, ds18b20_wo_bias, label='Promedio DS18B20 sin sesgo')
plt.plot(time_interp, ds18b20_linear_correction, label='Promedio DS18B20 con correción lineal')
plt.plot(time_interp, splined_calib1, label='Sensor calibrado 1')
plt.plot(time_interp, splined_calib2, label='Sensor calibrado 2')
plt.title("Temperatura corregida (ºC) vs tiempo (s)")
plt.legend()
plt.show()


'''Save linear correction parameters to .json file in workdir\output-data
so we can use them in polarimeter measurement:'''
parameters_filepath = "C:\\Users\\Admin\\rapp\\workdir\\output-data\\"
parameters_filename = "2024-11-14-temperature-correction-parameters.json"
parameters_file = os.path.join(parameters_filepath, parameters_filename)
comment = {
    'comment': 'Bias and linear correction parameters for DS18B20 sensor used in polarimeter room temperature '
               'measurements obtained from measurements with (calibrated sensors) the 14/11/2024'
}

correction_parameters = {
    'bias': '{}'.format(bias),
    'A': '{}'.format(linear_correction[0]),
    'b': '{}'.format(linear_correction[1])
}

json_data = {
    "comment": comment,
    "correction_parameters": correction_parameters
}

with open(parameters_file, 'w') as f:
    json.dump(json_data, f)

'''With this function we can load the parameters from the .json file:'''
with open(parameters_file, 'r') as f:
    json_data = json.load(f)

slope = json_data['correction_parameters']['A']
intercept = json_data['correction_parameters']['b']
print('A =', slope, 'b =', intercept)
