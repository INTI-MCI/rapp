import os

from rapp.measurement import Measurement

folder = 'data\\2024-05-02-simple-setup-drift-6000000'
filename = 'min-setup2-1-hwp0-cycles0-step45-samples6000000-rep1.txt'

filepath = os.path.join(folder, filename)
measurement = Measurement.from_file(filepath)
measurement._data.to_csv(filename[:-3] + "zip", compression='zip')
