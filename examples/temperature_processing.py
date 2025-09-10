import numpy as np
import pandas as pd
import json

TEMP_CORRECTION_FILE = r"C:\Users\Admin\rapp\examples\2025-06-19-test-temperature-correction-parameters.json"

with open(TEMP_CORRECTION_FILE, 'r') as f:
    json_data = json.load(f)

bias_0 = json_data['correction_parameters_sensor_0']['bias']
slope_0 = json_data['correction_parameters_sensor_0']['A']
intercept_0 = json_data['correction_parameters_sensor_0']['b']
print("Saved parameters for Sensor 0: ", 'A =', slope_0, 'b =', intercept_0)

bias_1 = json_data['correction_parameters_sensor_1']['bias']
slope_1 = json_data['correction_parameters_sensor_1']['A']
intercept_1 = json_data['correction_parameters_sensor_1']['b']
print("Saved parameters for Sensor 1: ", 'A =', slope_1, 'b =', intercept_1)

# 23/7/2025 with qp
filepath_0 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-23-test-temperature-measurement-cycles1.0-step1.0-samples134\temperature.csv"
filepath_1 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-23-test-temperature-measurement-cycles1.0-step1.0-samples134\qp-temperature.csv"

# 25/07/2025 no qp
filepath_0_nq_25 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-25-phase-diff-no-quartz-1-cycles1-step1-samples134\temperature.csv"
filepath_1_nq_25 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-25-phase-diff-no-quartz-1-cycles1-step1-samples134\qp-temperature.csv"

# 25/07/2025 with qp
filepath_0_wq_25 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-25-phase-diff-with-quartz-cycles1.0-step1.0-samples134\temperature.csv"
filepath_1_wq_25 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-25-phase-diff-with-quartz-cycles1.0-step1.0-samples134\qp-temperature.csv"

# 26/07/2025 no qp
filepath_0_nq_26 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-26-phase-diff-no-quartz-2-cycles1-step1-samples134\temperature.csv"
filepath_1_nq_26 = r"C:\Users\Admin\rapp\workdir\output-data\2025-07-26-phase-diff-no-quartz-2-cycles1-step1-samples134\qp-temperature.csv"


def correct_temperatures(temperature, a, b):
    a = float(a)
    b = float(b)
    return (((float(temperature) - b) / a) - b) / a


def correct_temperatures_bias(temperature, a, b, bias):
    bias = float(bias)
    a = float(a)
    b = float(b)
    measured_temperature = ((float(temperature) * a) + b) * a + b + bias
    return (measured_temperature - b) / a


def correct_temperatures_bias_raw(temperature, a, b, bias):
    bias = float(bias)
    a = float(a)
    b = float(b)
    measured_temperature = float(temperature) + bias
    return (measured_temperature - b) / a


def rewrite_temperature_file(filepath, sensor):
    if sensor == 0:
        bias = bias_0
        slope = slope_0
        intercept = intercept_0
    elif sensor == 1:
        bias = bias_1
        slope = slope_1
        intercept = intercept_1

    df = pd.read_csv(filepath, skiprows=2, header=None, delimiter=',')

    temperatures = df[1]
    print(df[1])

    for i in range(len(temperatures)):
        # temperatures[i] = round(correct_temperatures(temperatures[i], slope, intercept), 4)
        temperatures[i] = round(correct_temperatures_bias_raw(temperatures[i], slope, intercept, bias), 6)

    print(temperatures)
    df[1] = temperatures
    print(df)
    df.to_csv(filepath, sep=',', index=False, header=['ANGLE', 'TEMPERATURE', 'HWP-POS', 'REP'])

# rewrite_temperature_file(filepath_0_nq_25, sensor=0)


def process_temperatures(filepath1, filepath2):
    filepaths = [filepath1, filepath2]
    delta_temperatures = []
    avg_temperatures_channels = []
    max_temperatures = []
    min_temperatures = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, skiprows=2, header=None, delimiter=',')
        temperatures = df[1]
        avg_temperatures_channels.append(round(sum(temperatures) / len(temperatures), 6))
        delta_temperatures.append(max(temperatures) - min(temperatures))
        max_temperatures.append(max(temperatures))
        min_temperatures.append(min(temperatures))
    avg_temperature = np.mean(avg_temperatures_channels)
    total_delta_temperature = max(max_temperatures) - min(min_temperatures)
    return avg_temperature, delta_temperatures, total_delta_temperature

'''Medición 1, 27/08/2025'''
Sensor_0_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-08-27-phase-diff-no-quartz-1-cycles1-step1-samples134\temperature.csv"
Sensor_1_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-08-27-phase-diff-no-quartz-1-cycles1-step1-samples134\qp-temperature.csv"

Sensor_0_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-08-28-phase-diff-with-quartz-1-cycles1-step1-samples134\temperature.csv"
Sensor_1_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-08-28-phase-diff-with-quartz-1-cycles1-step1-samples134\qp-temperature.csv"

avg_temperature_no_qp, delta_temperatures_no_qp, total_delta_temperatures_no_qp = process_temperatures(Sensor_0_no_qp, Sensor_1_no_qp)
avg_temperature_w_qp, delta_temperatures_w_qp, total_delta_temperatures_w_qp = process_temperatures(Sensor_0_w_qp, Sensor_1_w_qp)
print("Medición 1, 27/08/2025")
print("T promedio sin placa: ", avg_temperature_no_qp, "DT: ", delta_temperatures_no_qp, "DT total: ", total_delta_temperatures_no_qp)
print("T promedio con placa: ", avg_temperature_w_qp, "DT: ", delta_temperatures_w_qp, "DT total: ", total_delta_temperatures_w_qp)
print("------------------")

'''Medición 2, 01/09/2025'''
Sensor_0_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-01-phase-diff-no-quartz-3reps-cycles1.0-step1.0-samples134\temperature.csv"
Sensor_1_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-01-phase-diff-no-quartz-3reps-cycles1.0-step1.0-samples134\qp-temperature.csv"

Sensor_0_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-01-phase-diff-with-quartz-3reps-cycles1.0-step1.0-samples134\temperature.csv"
Sensor_1_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-01-phase-diff-with-quartz-3reps-cycles1.0-step1.0-samples134\qp-temperature.csv"

avg_temperature_no_qp, delta_temperatures_no_qp, total_delta_temperatures_no_qp = process_temperatures(Sensor_0_no_qp, Sensor_1_no_qp)
avg_temperature_w_qp, delta_temperatures_w_qp, total_delta_temperatures_w_qp = process_temperatures(Sensor_0_w_qp, Sensor_1_w_qp)
print("Medición 2, 01/09/2025")
print("T promedio sin placa: ", avg_temperature_no_qp, "DT: ", delta_temperatures_no_qp, "DT total: ", total_delta_temperatures_no_qp)
print("T promedio con placa: ", avg_temperature_w_qp, "DT: ", delta_temperatures_w_qp, "DT total: ", total_delta_temperatures_w_qp)
print("------------------")

'''Medición 3, 05/09/2025'''
Sensor_0_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-05-phase-diff-no-quartz-10reps-FNCT-cycles1.0-step10.0-samples134\temperature.csv"
Sensor_1_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-05-phase-diff-no-quartz-10reps-FNCT-cycles1.0-step10.0-samples134\qp-temperature.csv"

Sensor_0_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-05-phase-diff-with-quartz-10reps-FNCT-cycles1.0-step10.0-samples134\temperature.csv"
Sensor_1_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-05-phase-diff-with-quartz-10reps-FNCT-cycles1.0-step10.0-samples134\qp-temperature.csv"

avg_temperature_no_qp, delta_temperatures_no_qp, total_delta_temperatures_no_qp = process_temperatures(Sensor_0_no_qp, Sensor_1_no_qp)
avg_temperature_w_qp, delta_temperatures_w_qp, total_delta_temperatures_w_qp = process_temperatures(Sensor_0_w_qp, Sensor_1_w_qp)
print("Medición 3, 05/09/2025")
print("T promedio sin placa: ", avg_temperature_no_qp, "DT: ", delta_temperatures_no_qp, "DT total: ", total_delta_temperatures_no_qp)
print("T promedio con placa: ", avg_temperature_w_qp, "DT: ", delta_temperatures_w_qp, "DT total: ", total_delta_temperatures_w_qp)
print("------------------")

'''Medición 4, 08/09/2025'''
Sensor_0_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-08-phase-diff-no-quartz-10reps-FNCT-2da-tirada-cycles1.0-step5.0-samples134\temperature.csv"
Sensor_1_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-08-phase-diff-no-quartz-10reps-FNCT-2da-tirada-cycles1.0-step5.0-samples134\qp-temperature.csv"

Sensor_0_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-08-phase-diff-with-quartz-10reps-FNCT-2da-tirada-cycles1.0-step5.0-samples134\temperature.csv"
Sensor_1_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-08-phase-diff-with-quartz-10reps-FNCT-2da-tirada-cycles1.0-step5.0-samples134\qp-temperature.csv"

avg_temperature_no_qp, delta_temperatures_no_qp, total_delta_temperatures_no_qp = process_temperatures(Sensor_0_no_qp, Sensor_1_no_qp)
avg_temperature_w_qp, delta_temperatures_w_qp, total_delta_temperatures_w_qp = process_temperatures(Sensor_0_w_qp, Sensor_1_w_qp)
print("Medición 4, 08/09/2025")
print("T promedio sin placa: ", avg_temperature_no_qp, "DT: ", delta_temperatures_no_qp, "DT total: ", total_delta_temperatures_no_qp)
print("T promedio con placa: ", avg_temperature_w_qp, "DT: ", delta_temperatures_w_qp, "DT total: ", total_delta_temperatures_w_qp)
print("------------------")

'''Medición 5, 09/09/2025'''
Sensor_0_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-09-phase-diff-no-quartz-7reps-FNCT-cycles1.0-step5.0-samples134\temperature.csv"
Sensor_1_no_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-09-phase-diff-no-quartz-7reps-FNCT-cycles1.0-step5.0-samples134\qp-temperature.csv"

Sensor_0_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-09-phase-diff-with-quartz-7reps-FNCT-cycles1.0-step5.0-samples134\temperature.csv"
Sensor_1_w_qp = r"C:\Users\Admin\rapp\workdir\output-data\2025-09-09-phase-diff-with-quartz-7reps-FNCT-cycles1.0-step5.0-samples134\qp-temperature.csv"

avg_temperature_no_qp, delta_temperatures_no_qp, total_delta_temperatures_no_qp = process_temperatures(Sensor_0_no_qp, Sensor_1_no_qp)
avg_temperature_w_qp, delta_temperatures_w_qp, total_delta_temperatures_w_qp = process_temperatures(Sensor_0_w_qp, Sensor_1_w_qp)
print("Medición 5, 09/09/2025")
print("T promedio sin placa: ", avg_temperature_no_qp, "DT: ", delta_temperatures_no_qp, "DT total: ", total_delta_temperatures_no_qp)
print("T promedio con placa: ", avg_temperature_w_qp, "DT: ", delta_temperatures_w_qp, "DT total: ", total_delta_temperatures_w_qp)
print("------------------")

'''Gráfico del poster, diferencia de fase sin placa, entre líneas 216 y 217 de rapp/analysis/phase_diff.py:
# TODO: agregar barras de error a las diferencias de fase
        plt.figure()
        plt.plot(phase_diffs, ".-", color="k", label=label_phase_diff)
        plt.ylabel("Diferencia de fase (°)", size=14)
        plt.xlabel("Nº de repetición", size=14)
        plt.title("Diferencia de fase sin placa", size=17)
        twin2 = plt.twinx()
        twin2.plot(temperature0[0], linestyle="-", color="r", label="Temperatura Media 0")
        twin2.set_ylabel("Temperatura (°C)", size=14)
        twin2.plot(temperature0[1], linestyle=":", color="r", label="Temperatura Max/Min 0")
        twin2.plot(temperature0[2], linestyle=":", color="r")
        twin2.plot(temperature1[0], linestyle="-", color="b", label="Temperatura Media 1")
        # twin2.set_ylabel("Temperatura (°C)")
        twin2.plot(temperature1[1], linestyle=":", color="b", label="Temperatura Max/Min 1")
        twin2.plot(temperature1[2], linestyle=":", color="b")
        twin2.legend()
        axs[2].legend()

        f.tight_layout()

'''
