from rapp.signal.analysis import read_measurement_file

filepath = 'data/29-12-2023/hwp0/sine-hwp0-rep1-cycles1-step1.0-samples50.txt'

data = read_measurement_file(filepath, usecols=None)

print(data)

ch0 = data['CH0']
ch1 = data['CH1']

data['CH0'] = ch1
data['CH1'] = ch0

data.to_csv(filepath, sep='\t', index=False)
