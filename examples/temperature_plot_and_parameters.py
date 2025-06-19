import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy import odr
import os
import json

''' 
This example provides correction parameters for the temperature sensor used 
in polarimeter measurements (DS18B20 sensor).
The data used consists of the temperature measured by the DS18B20 sensor (filepath1) 
and by two calibrated temperature sensors (filepath2) during a certain period of time.
If there is more than one DS18B20 sensor, the data is collected from all of them for 
further analysis (see N_SENSORS_DS18B20 and filepaths).
The temperature measurements are sampled simultaneously, but at different times, 
so the signals were interpolated using a single time axis for further analysis.
We chose to analyze the bias correction and a linear correction for the DS18B20 sensor. 
'''

N_SENSORS_DS18B20 = 2

# filepath1 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\2025-05-28-1748456467.473517-temperatura-sensor-1-tiempo-total-600-tiempo-espera-120-muestras10-.txt'
# filepath3 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\2025-05-28-1748456467.4731185-temperatura-sensor-0-tiempo-total-600-tiempo-espera-120-muestras10-.txt'
if N_SENSORS_DS18B20 == 1:
    filepath1 = r'C:\Users\Admin\rapp\workdir\output-data\2024-11-14-1731605585.2825165-temperatura-tiempo-total-9000-tiempo-espera-120-muestras10.txt'
    filepath2 = r'C:\Users\Admin\rapp\workdir\output-data\2024-11-14-mediciones-temp-sensores-calibrados.txt'
if N_SENSORS_DS18B20 == 2:
    # filepath1 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\2025-05-26-temperatura-sensor-0.txt'
    filepath1 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\2025-05-29-1748544704.2162209-temperatura\sensor-0-tiempo-total-57600-tiempo-espera-120-muestras-10.txt'
    # filepath2 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\keithley.txt'
    filepath2 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\2025-05-29-keithley.txt'
    # filepath3 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\2025-05-26-temperatura-sensor-1.txt'
    filepath3 = r'C:\Users\cvargas\Documents\Mediciones\comparacion-ds18b20-keithley\2025-05-29-1748544704.2162209-temperatura\sensor-1-tiempo-total-57600-tiempo-espera-120-muestras-10.txt'

def time_to_seconds(time_str) -> int:
    h, m, s = map(int, time_str.split(':'))
    return h*3600 + m*60 + s


def add_date_to_timestamps(timestamps):
    # Inicializa la fecha actual (asumimos que es el primer día)
    fecha_actual = datetime(2022, 1, 1)
    fecha_hora = []

    cambio_de_dia = True
    for timestamp in timestamps:
        # Convierte el timestamp a un objeto datetime
        hora_actual = datetime.strptime(timestamp, '%H:%M:%S')

        # Combina la fecha actual con la hora actual
        fecha_hora_actual = fecha_actual.replace(hour=hora_actual.hour, minute=hora_actual.minute,
                                                 second=hora_actual.second)
        # print(f"Fecha y hora actual: {fecha_hora_actual.strftime('%Y-%m-%d %H:%M:%S')}")

        # Le agregamos un día a las mediciones hechas entre las 00:00 y las 08:59
        if hora_actual.hour < 9 and cambio_de_dia:
            fecha_actual += timedelta(days=1)
            # print(f"Cambio de día: {fecha_actual.strftime('%Y-%m-%d')}")
            cambio_de_dia = False
        fecha_hora.append(
            fecha_hora_actual.replace(year=fecha_actual.year, month=fecha_actual.month, day=fecha_actual.day))
        # print(f"Fecha y hora actual despues del replace: {fecha.strftime('%Y-%m-%d %H:%M:%S')}")

    # time_ds18b20_0 = np.array([f.strftime('%Y-%m-%d %H:%M:%S') for f in fecha_hora])

    timestamps_array = np.array([(f.timestamp()) / 10 ** 9 for f in fecha_hora])
    return timestamps_array


def u_temperature_ds18b20(value):
    delta_res = 0.0625
    ures = delta_res / np.sqrt(12)
    uncertainty = np.sqrt((np.std(value, axis=1) / np.sqrt(10)) ** 2 + ures ** 2)
    return uncertainty


def weighted_average(value, uvalue):
    weight = 1 / uvalue**2
    avg = np.sum(value * weight) / np.sum(weight)
    uavg = np.sqrt(1 / np.sum(weight))
    return avg, uavg


def bias_correction(splined_ds18b20, u_ds18b20, splined_calib1, splined_calib2):
    """Bias correction for DS18B20 measurements using the interpolated signals:"""
    bias_1_signal = splined_ds18b20 - splined_calib1
    bias_1, ubias_1 = weighted_average(bias_1_signal, u_ds18b20)
    # bias_1 = np.mean(bias_1_signal)
    # ubias_1 = np.std(bias_1_signal) / np.sqrt(len(bias_1_signal))

    bias_2_signal = splined_ds18b20 - splined_calib2
    bias_2, ubias_2 = weighted_average(bias_2_signal, u_ds18b20)
    # bias_2 = np.mean(bias_2_signal)
    # ubias_2 = np.std(bias_2_signal) / np.sqrt(len(bias_2_signal))

    bias = np.mean((bias_1, bias_2))
    ubias = np.sqrt(ubias_1**2 + ubias_2**2) / 2
    ds18b20_wo_bias = splined_ds18b20 - bias

    return ds18b20_wo_bias, bias, ubias


def rmse(signal, mean_signal):
    m = len(signal)
    return np.linalg.norm(signal - mean_signal) / np.sqrt(m)


def linear(x, a, b):
    return a * x + b


def linear_odr(beta, x):
    return beta[0] * x + beta[1]


'''Import ds18b20 data from sensor 0 and calculate average and std:'''
data1_ = np.genfromtxt(filepath1, delimiter=',', skip_header=1, dtype=str)
data1 = np.array(data1_.tolist())

data_DS18B20_0 = data1[:, :9].astype(float)
N = data1.shape[0]

timestamps_ds18b20_0 = data1[:, 10]
time_ds18b20_0 = add_date_to_timestamps(timestamps_ds18b20_0)
average_ds18b20_0 = np.mean(data_DS18B20_0, axis=1)
print(f"Promedio del sensor 0 = {np.mean(average_ds18b20_0):.3f}")
av_ds18b20_wo_mean_0 = average_ds18b20_0 - np.mean(average_ds18b20_0)
stds_ds18b20_0 = np.std(data_DS18B20_0, axis=1)
print(f"Desviacion estandar del sensor 0 = {np.mean(stds_ds18b20_0):.3f}")
print(average_ds18b20_0.shape, stds_ds18b20_0.shape)
count_std_zero = np.count_nonzero(stds_ds18b20_0 == 0)
print(f"{count_std_zero=}/{N}")
pooled_std_0 = np.sqrt(np.sum(stds_ds18b20_0**2) / N)
u_ds18b20_0 = u_temperature_ds18b20(data_DS18B20_0)
#TODO:
print('Desviacion combinada del sensor 0 =', pooled_std_0)

print('Promedio de las dispersiones del sensor 0 =', np.mean(stds_ds18b20_0))
print('Promedio de las incertidumbres del sensor 0 =', np.mean(u_ds18b20_0))


if N_SENSORS_DS18B20 == 2:
    '''Import ds18b20 data from sensor 1 and calculate average and std:'''
    data3_ = np.genfromtxt(filepath3, delimiter=',', skip_header=1, dtype=str)
    data3 = np.array(data3_.tolist())
    data_DS18B20_1 = data3[:, :10].astype(float)
    assert N == data3.shape[0]

    timestamps_ds18b20_1 = data3[:, 10]
    time_ds18b20_1 = add_date_to_timestamps(timestamps_ds18b20_1)

    average_ds18b20_1 = np.mean(data_DS18B20_1, axis=1)
    av_ds18b20_wo_mean_1 = average_ds18b20_1 - np.mean(average_ds18b20_1)
    stds_ds18b20_1 = np.std(data_DS18B20_1, axis=1)
    pooled_std_1 = np.sqrt(np.sum(stds_ds18b20_1**2) / N)
    u_ds18b20_1 = u_temperature_ds18b20(data_DS18B20_1)
    print('Desviacion combinada del sensor 1 =', pooled_std_1)
    print('Promedio de las dispersiones del sensor 1 =', np.mean(stds_ds18b20_1))
    print('Promedio de las incertidumbres del sensor 1 =', np.mean(u_ds18b20_1))


'''Import calibrated sensors data:'''
data2 = np.genfromtxt(filepath2, delimiter=',')
# skip_header=17, skip_footer=16) (para las mediciones del 7-11)
sensor1 = data2[:, 0]
sensor1_wo_mean = sensor1 - np.mean(sensor1)

sensor2 = data2[:, 1]
sensor2_wo_mean = sensor2 - np.mean(sensor2)
if N_SENSORS_DS18B20 == 1:
    time2 = np.loadtxt(filepath2, dtype=str, usecols=3, delimiter=',')

# time_calibrated_sensors = np.array([datetime.strptime(t, '%H:%M:%S') for t in time2])

if N_SENSORS_DS18B20 == 2:
    time2 = np.loadtxt(filepath2, dtype=str, usecols=3, delimiter=',')
time_calibrated_sensors = add_date_to_timestamps(time2)

u_calibrated_sensors = 0.03

'''Plot:'''
plt.errorbar(time_ds18b20_0, average_ds18b20_0, yerr=stds_ds18b20_0/np.sqrt(10), label='Promedio 0 DS18B20')
if N_SENSORS_DS18B20 == 2:
    plt.errorbar(time_ds18b20_1, average_ds18b20_1, yerr=stds_ds18b20_1/np.sqrt(10), label='Promedio 1 DS18B20')
plt.plot(time_calibrated_sensors, sensor1, label='Sensor calibrado 1')
plt.plot(time_calibrated_sensors, sensor2, label='Sensor calibrado 2')
plt.title("Mediciones temperatura (ºC) vs tiempo")
plt.legend()
plt.show()

'''Interpolated signals:'''

if N_SENSORS_DS18B20 == 1:
    min_time_range = max(time_ds18b20_0[0], time_calibrated_sensors[0])
    max_time_range = min(time_ds18b20_0[-1], time_calibrated_sensors[-1])

elif N_SENSORS_DS18B20 == 2:
    min_time_range = max(time_ds18b20_0[0], time_ds18b20_1[0], time_calibrated_sensors[0])
    max_time_range = min(time_ds18b20_0[-1], time_ds18b20_1[-1], time_calibrated_sensors[-1])
time_interp = np.linspace(min_time_range, max_time_range, N)
splined_ds18b20_0 = np.interp(time_interp, time_ds18b20_0, average_ds18b20_0)
if N_SENSORS_DS18B20 == 2:
    splined_ds18b20_1 = np.interp(time_interp, time_ds18b20_1, average_ds18b20_1)

splined_calib1 = np.interp(time_interp, time_calibrated_sensors, sensor1)
splined_calib2 = np.interp(time_interp, time_calibrated_sensors, sensor2)
splined_calib_mean = (splined_calib1 + splined_calib2) / 2
plt.plot(time_interp, splined_ds18b20_0, label='Promedio 0 DS18B20')
if N_SENSORS_DS18B20 == 2:
    plt.plot(time_interp, splined_ds18b20_1, label='Promedio 1 DS18B20')
plt.plot(time_interp, splined_calib1, label='Sensor calibrado 1')
plt.plot(time_interp, splined_calib2, label='Sensor calibrado 2')
plt.title("Temperatura interpolada (ºC) vs tiempo (s)")
plt.legend()
plt.show()

'''Bias correction for DS18B20 measurements using the interpolated signals:'''
ds18b20_0_wo_bias, bias_sensor_0, ubias_sensor_0 = bias_correction(splined_ds18b20_0, u_ds18b20_0, splined_calib1, splined_calib2)
print(f"Bias for sensor 0: {bias_sensor_0:.5f} +- {ubias_sensor_0:.5f}")
RMSE_bias_sensor_0 = rmse(ds18b20_0_wo_bias, splined_calib_mean)
print(f"RMSE for bias sensor 0: {RMSE_bias_sensor_0:.5f}")

if N_SENSORS_DS18B20 == 2:
    ds18b20_1_wo_bias, bias_sensor_1, ubias_sensor_1 = bias_correction(splined_ds18b20_1, u_ds18b20_1, splined_calib1, splined_calib2)
    print(f"Bias for sensor 1: {bias_sensor_1:.5f} +- {ubias_sensor_1:.5f}")
    RMSE_bias_sensor_1 = rmse(ds18b20_1_wo_bias, splined_calib_mean)
    print(f"RMSE for bias sensor 1: {RMSE_bias_sensor_1:.5f}")


''' Linear correction for DS18B20 measurements using the interpolated signals:'''
# Sensor 0:
A_ds18b20_0 = np.hstack([splined_ds18b20_0[:, np.newaxis], np.ones((N, 1))])
A1_0 = np.linalg.inv(A_ds18b20_0.T @ A_ds18b20_0) @ A_ds18b20_0.T @ splined_calib1[:, np.newaxis]

A2_0 = np.linalg.inv(A_ds18b20_0.T @ A_ds18b20_0) @ A_ds18b20_0.T @ splined_calib2[:, np.newaxis]
linear_correction_sensor_0 = (A1_0 + A2_0) / 2

print('Linear correction parameters for sensor 0: [[A] [b]] =', linear_correction_sensor_0)

ds18b20_linear_correction_0 = A_ds18b20_0 @ linear_correction_sensor_0

# A_sensor_0 = np.linalg.inv(A_ds18b20_0.T @ A_ds18b20_0) @ A_ds18b20_0.T @ splined_calib_mean[:, np.newaxis]
# print('Linear correction parameters for sensor 0 from mean signal: [[A] [b]] =', A_sensor_0)

error_0 = rmse(ds18b20_linear_correction_0.flatten(), splined_calib_mean)
print('RMSE sensor 0 =', error_0)

# def cost_function(linear_correction_parameters, ds18b20, calibrated_sensors):
#     A_ds18b20 = np.hstack([ds18b20[:, np.newaxis], np.ones((N, 1))])
#     J =

"""Linear correction using curve_fit: """
popt0, pcov0 = curve_fit(linear, xdata = splined_calib_mean, ydata = splined_ds18b20_0, sigma=u_ds18b20_0, absolute_sigma=True)
perr0 = np.sqrt(np.diag(pcov0))
print('Linear correction parameters for sensor 0 from curve_fit: [[A] [b]] =', popt0)
print('Standard errors for linear correction parameters (sensor 0) from curve_fit:', perr0)

ds18b20_linear_correction_curve_fit_0 = (splined_ds18b20_0 - popt0[1]) / popt0[0]

"""Linear correction using odr: """
model_odr = odr.Model(fcn=linear_odr, estimate=np.array([1, 0]))
data_odr_0 = odr.Data(x=splined_calib_mean, y=splined_ds18b20_0, we=1/(u_ds18b20_0**2), wd=1/(0.03**2))

odr0 = odr.ODR(data_odr_0, model_odr, beta0=np.array([1, 0]))
odr0.set_job(fit_type=0, deriv=1)

output_odr0 = odr0.run()

print('Linear correction parameters for sensor 0 from odr: [[A] [b]] =', output_odr0.beta)
print('Standard errors for linear correction parameters (sensor 0) from odr:', output_odr0.sd_beta)

ds18b20_linear_correction_odr_0 = (splined_ds18b20_0 - output_odr0.beta[1]) / output_odr0.beta[0]

if N_SENSORS_DS18B20 == 2:
    # Sensor 1:
    A_ds18b20_1 = np.hstack([splined_ds18b20_1[:, np.newaxis], np.ones((N, 1))])

    A1_1 = np.linalg.inv(A_ds18b20_1.T @ A_ds18b20_1) @ A_ds18b20_1.T @ splined_calib1[:, np.newaxis]
    A2_1 = np.linalg.inv(A_ds18b20_1.T @ A_ds18b20_1) @ A_ds18b20_1.T @ splined_calib2[:, np.newaxis]

    linear_correction_sensor_1 = (A1_1 + A2_1) / 2
    print(linear_correction_sensor_1[0], linear_correction_sensor_1[1])
    print('Linear correction parameters for sensor 1: [[A] [b]] =', linear_correction_sensor_1)

    ds18b20_linear_correction_1 = A_ds18b20_1 @ linear_correction_sensor_1


    # A_sensor_1 = np.linalg.inv(A_ds18b20_1.T @ A_ds18b20_1) @ A_ds18b20_1.T @ splined_calib_mean[:, np.newaxis]
    # print('Linear correction parameters for sensor 1 from mean signal: [[A] [b]] =', A_sensor_1)

    error_1_signal = ds18b20_linear_correction_1.flatten() - splined_calib_mean
    error_1 = np.linalg.norm(error_1_signal) / np.sqrt(N)
    print('RMSE sensor 1=', error_1)

    popt1, pcov1 = curve_fit(linear, xdata=splined_calib_mean, ydata=splined_ds18b20_1, sigma=u_ds18b20_1, absolute_sigma=True)
    perr1 = np.sqrt(np.diag(pcov1))
    print('Linear correction parameters for sensor 1 from curve_fit: [[A] [b]] =', popt1)
    print('Standard errors for linear correction parameters (sensor 1) from curve_fit:', perr1)

    ds18b20_linear_correction_curve_fit_1 = (splined_ds18b20_1 - popt1[1]) / popt1[0]

    data_odr_1 = odr.Data(x=splined_calib_mean, y=splined_ds18b20_1, we=1/(u_ds18b20_1**2), wd=1/(0.03**2))

    odr1 = odr.ODR(data_odr_1, model_odr, beta0=np.array([1, 0]))
    odr1.set_job(fit_type=0, deriv=1)

    output_odr1 = odr1.run()

    print('Linear correction parameters for sensor 1 from odr: [[A] [b]] =', output_odr1.beta)
    print('Standard errors for linear correction parameters (sensor 1) from odr:', output_odr1.sd_beta)

    ds18b20_linear_correction_odr_1 = (splined_ds18b20_1 - output_odr1.beta[1]) / output_odr1.beta[0]

plt.plot(time_interp, ds18b20_0_wo_bias, label='Promedio DS18B20 0 sin sesgo')
if N_SENSORS_DS18B20 == 2:
    plt.plot(time_interp, ds18b20_1_wo_bias, label='Promedio DS18B20 1 sin sesgo')
plt.plot(time_interp, ds18b20_linear_correction_0, label='Promedio DS18B20 0 con correción lineal')
if N_SENSORS_DS18B20 == 2:
    plt.plot(time_interp, ds18b20_linear_correction_1, label='Promedio DS18B20 1 con correción lineal')
plt.plot(time_interp, splined_calib1, label='Sensor calibrado 1')
plt.plot(time_interp, splined_calib2, label='Sensor calibrado 2')
plt.title("Temperatura corregida (ºC) vs tiempo (s)")
plt.legend()
plt.show()

plt.figure()
plt.plot(time_interp, ds18b20_0_wo_bias, label='Promedio DS18B20 0 sin sesgo')
plt.plot(time_interp, ds18b20_linear_correction_0, label='Promedio DS18B20 0 con correción lineal')
plt.plot(time_interp, ds18b20_linear_correction_curve_fit_0, label='Promedio DS18B20 0 con curve_fit')
plt.plot(time_interp, ds18b20_linear_correction_odr_0, label='Promedio DS18B20 0 con odr')
if N_SENSORS_DS18B20 == 2:
    plt.plot(time_interp, ds18b20_1_wo_bias, label='Promedio DS18B20 1 sin sesgo')
    plt.plot(time_interp, ds18b20_linear_correction_1, label='Promedio DS18B20 1 con correción lineal')
    plt.plot(time_interp, ds18b20_linear_correction_curve_fit_1, label='Promedio DS18B20 1 con curve_fit')
    plt.plot(time_interp, ds18b20_linear_correction_odr_1, label='Promedio DS18B20 1 con odr')
plt.plot(time_interp, splined_calib1, label='Sensor calibrado 1')
plt.plot(time_interp, splined_calib2, label='Sensor calibrado 2')
plt.title("Temperatura corregida (ºC) vs tiempo (s)")
plt.legend()
plt.show()

plt.figure()
plt.plot(ds18b20_linear_correction_0.flatten() - splined_calib_mean, label='DS18B20 0')
plt.plot(ds18b20_linear_correction_1.flatten() - splined_calib_mean, label='DS18B20 1')
plt.title("Temperatura corregida - promedio sensores calibrados (ºC)")
plt.legend()
plt.show()

plt.figure()
plt.plot(splined_calib_mean, splined_ds18b20_0, '.', label='Promedio DS18B20 0')
plt.plot(splined_calib_mean, (splined_calib_mean - linear_correction_sensor_0[1][0]) / linear_correction_sensor_0[0][0], label='Correción lineal DS18B20 0')
plt.plot(splined_calib_mean, popt0[0] * splined_calib_mean + popt0[1], label='Correción lineal con curve_fit DS18B20 0')
plt.plot(splined_calib_mean, output_odr0.beta[0] * splined_calib_mean + output_odr0.beta[1], label='Correción lineal con odr DS18B20 0')
if N_SENSORS_DS18B20 == 2:
    plt.plot(splined_calib_mean, splined_ds18b20_1, '.', label='Promedio DS18B20 1')
    plt.plot(splined_calib_mean, (splined_calib_mean - linear_correction_sensor_1[1][0]) / linear_correction_sensor_1[0][0], label='Correción lineal DS18B20 1')
    plt.plot(splined_calib_mean, popt1[0] * splined_calib_mean + popt1[1], label='Correción lineal con curve_fit DS18B20 1')
    plt.plot(splined_calib_mean, output_odr1.beta[0] * splined_calib_mean + output_odr1.beta[1], label='Correción lineal con odr DS18B20 1')

plt.title("Temperatura DS18B20 vs temperatura promedio sensores calibrados (ºC)")
plt.legend()
plt.show()


'''Save linear correction parameters to .json file in workdir\output-data
so we can use them in polarimeter measurement:'''
parameters_filepath = "C:\\Users\\Admin\\rapp\\workdir\\output-data\\"
parameters_filename = "2025-05-26-test-temperature-correction-parameters.json"
parameters_file = os.path.join(parameters_filepath, parameters_filename)
comment = {
    'comment': 'Bias and linear correction parameters for DS18B20 sensors used in polarimeter room temperature '
               'measurements obtained from measurements with (calibrated sensors) the 26/05/2025'
}

correction_parameters_0 = {
    'bias': '{}'.format(bias_sensor_0),
    'A': '{}'.format(linear_correction_sensor_0[0][0]),
    'b': '{}'.format(linear_correction_sensor_0[1][0])
}

if N_SENSORS_DS18B20 == 2:
    correction_parameters_1 = {
        'bias': '{}'.format(bias_sensor_1),
        'A': '{}'.format(linear_correction_sensor_1[0][0]),
        'b': '{}'.format(linear_correction_sensor_1[1][0])
    }
if N_SENSORS_DS18B20 == 1:
    json_data = {
        "comment": comment,
        "correction_parameters_sensor_0": correction_parameters_0
    }
if N_SENSORS_DS18B20 == 2:
    json_data = {
        "comment": comment,
        "correction_parameters_sensor_0": correction_parameters_0,
        "correction_parameters_sensor_1": correction_parameters_1
    }

with open(parameters_file, 'w') as f:
    json.dump(json_data, f)

'''With this function we can load the parameters from the .json file:'''
with open(parameters_file, 'r') as f:
    json_data = json.load(f)

slope_0 = json_data['correction_parameters_sensor_0']['A']
intercept_0 = json_data['correction_parameters_sensor_0']['b']
print("Sensor 0: ", 'A =', slope_0, 'b =', intercept_0)

if N_SENSORS_DS18B20 == 2:
    slope_1 = json_data['correction_parameters_sensor_1']['A']
    intercept_1 = json_data['correction_parameters_sensor_1']['b']
    print("Sensor 1: ", 'A =', slope_1, 'b =', intercept_1)
