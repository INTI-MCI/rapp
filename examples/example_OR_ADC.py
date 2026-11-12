import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Usuario\\rapp\\data\\table-riao-final.csv", sep="\t", header=0)
# phase_diff = data['DIFF MEAN'].to_numpy()
# optical_rotation = []

# for i in range(12):
#     indexes = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
#     o_r = (phase_diff[indexes[i]:indexes[i]+7] - phase_diff[indexes[i]])/2
#     optical_rotation = optical_rotation + list(o_r)

# data['OR'] = optical_rotation
# data.to_csv("C:\\Users\\Usuario\\rapp\\data\\table-riao-final.csv", sep='\t', index=False)

plt.rcParams.update({'font.size': 16})

plt.figure()
plt.scatter(data['HWP'][1:7], -data['OR'][1:7], marker='o', label='CS1237')
plt.scatter(data['HWP'][29:35], -data['OR'][29:35], marker='o', label='ADS1115')
plt.scatter(data['HWP'][57:63], -data['OR'][57:63], marker='o', label='ADS1232')
plt.plot(data['HWP'][1:7], data['HWP'][1:7], linestyle='-.', color='gray')
plt.xlabel('Half Wave Plate Optical Rotation')
plt.ylabel('Measured Optical Rotation')
# plt.title('Optical Rotation')
plt.legend()
plt.show()

plt.figure()
plt.plot(data['HWP'][15:21], -data['OR'][15:21], marker='o', linestyle='-', label='CS1237')
plt.plot(data['HWP'][43:49], -data['OR'][43:49], marker='o', linestyle='-', label='ADS1115')
plt.plot(data['HWP'][71:77], -data['OR'][71:77], marker='o', linestyle='-', label='ADS1232')
plt.xlabel('HWP OR')
plt.ylabel('Measured Optical Rotation')
# plt.title('Optical Rotation')
plt.legend()
plt.show()
