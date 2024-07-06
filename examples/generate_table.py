import os
import logging

from rich.progress import track

from rapp.analysis import phase_diff
from rapp.data_file import DataFile


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)

COLUMN_NAMES = [
    "CYCLES", "STEP", "SAMPLES", "REPS",
    "CH0 MEAN", "CH0 STD", "CH1 MEAN", "CH1 STD", "DIFF MEAN", "DIFF STD", "DIFF STD STD"
]

FOLDERS = [
    "2024-03-05-repeatability/hwp0",
    "2024-03-05-repeatability/hwp9",
    "2024-03-26-no-hwp",
    "2024-05-20-full-setup-search-for-optimum-cycles1-step1-samples169",
    "2024-05-21-full-setup-search-for-optimum-cycles2-step1-samples169",
    "2024-05-22-full-setup-search-for-optimim-cycles1-step0.5-samples169",
    "2024-05-23-cuarzo-sucio-full-setup",
    "2024-05-23-full-setup-search-for-optimim-cycles1-step1-samples1268",
    "2024-05-24-full-setup-norm-exp1-cycles1-step1-samples169",
    "2024-05-24-full-setup-norm-exp2-cycles2-step1-samples169",
    "2024-05-24-full-setup-norm-exp3-cycles1-step0.5-samples169",
    "2024-05-24-full-setup-norm-exp4-cycles1-step1-samples1218",
    "2024-05-24-full-setup-norm-exp5-cycles2-step2-samples1218",
    "2024-05-31-full-setup-norm-OD2-exp1-cycles1-step1-samples169",
    "2024-05-31-full-setup-norm-OD2-exp2-cycles2-step1-samples169",
    "2024-05-31-full-setup-norm-OD2-exp3-cycles1-step0.5-samples169",
    "2024-05-31-full-setup-norm-OD2-exp4-cycles1-step1-samples1218",
    "2024-05-31-full-setup-norm-OD2-exp5-cycles2-step2-samples1218",
    "2024-06-04-full-setup-norm-quartz-plate-cycles1-step1-samples169",
    "2024-06-07-full-setup-norm-1-hwp0-cycles1-step1-samples169",
    "2024-06-07-full-setup-norm-2-hwp0-cycles4-step1-samples169",
    "2024-06-07-full-setup-norm-3-hwp0-cycles1-step0.25-samples169",
    "2024-06-10-full-setup-norm-hwp9.8-cycles1-step0.25-samples169",
    "2024-06-12-full-quartz-1-cycles1-step1-samples169",
    "2024-06-12-full-quartz-2-cycles3-step1-samples169",
    "2024-06-12-full-quartz-3-cycles1-step1-samples1218",
    "2024-06-13-full-quartz-1-cycles1-step1-samples169",
    "2024-06-13-full-quartz-2-cycles1-step1-samples1218",
    "2024-06-14-quartz-velocities-1-cycles1-step1-samples169",
    "2024-06-15-quartz-velocities-2-cycles1-step1-samples169",
    "2024-06-15-quartz-velocities-3-cycles1-step1-samples169",
    "2024-06-15-quartz-velocities-4-cycles1-step1-samples169",
    "2024-06-16-quartz-velocities-5-cycles1-step1-samples169",
    "2024-06-25-quartz-vel-1-acc-0.5-deacc-0.5-1-cycles1-step1-samples169"
]


def setup_logger():
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)

    # Hide logging from external libraries
    external_libs = ['matplotlib', 'PIL']
    for lib in external_libs:
        logging.getLogger(lib).setLevel(logging.ERROR)


def main():
    table_file = DataFile(column_names=COLUMN_NAMES, delimiter='\t')
    table_file.open("table.csv")

    for folder in track(FOLDERS, description="Generating table..."):
        folder = os.path.join("data", folder)
        row_nls = phase_diff.phase_difference_from_folder(folder, method='NLS')
        row_dft = phase_diff.phase_difference_from_folder(folder, method='DFT')[4:]
        row = row_nls + row_dft
        table_file.add_row(row)

    table_file.close()


if __name__ == '__main__':
    main()
