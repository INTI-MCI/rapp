from rapp.signal.analysis import read_measurement_file

filepath = r'data\28-12-2023\hwp29\sine-hwp29-rep3-cycles1-step1.0-samples50.txt'

data = read_measurement_file(filepath, usecols=None)

print(data)

ch0 = data['CH0']
ch1 = data['CH1']

data['CH0'] = ch1
data['CH1'] = ch0

data.to_csv(filepath, sep='\t', index=False)
