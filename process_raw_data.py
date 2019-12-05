'''
Process the raw data CSV to compute the SWL scores from the satisfaction survey responses.
'''

import csv
import copy
import argparse
from pathlib import Path
from utils import init_logger

logger = init_logger()

RESPONE_SCORE_MAP = {
    'strongly agree': 7,
    'agree': 6,
    'slightly agree': 5,
    'neither agree nor disagree': 4,
    'slightly disagree': 3,
    'disagree': 2,
    'strongly disagree': 1
}

parser = argparse.ArgumentParser(description='Process the raw data CSV to compute the SWL scores from the satisfaction survey responses.')
parser.add_argument('input', type=str, help='The path to the raw dat CSV file.')
parser.add_argument('--output', type=str, help='The filepath of the output file.', default=None)
parser.add_argument('-q', '--swl-questions', nargs='+', type=int, help='The (zero-based) indices of the columns representing the responses to the SWL satsification survey.')
parser.add_argument('--swl-header-name', type=str, help='The name of the SWL score header in the processed CSV', default='SWL Score')
args = parser.parse_args()

input_path = Path(args.input)
if not (input_path.is_file() or input_path.exists()):
    logger.error('The specified input is not a file or does not exist!')
    exit(1)

if args.output is None:
    output_path = input_path.with_suffix('.processed' + input_path.suffix)
else:
    output_path = Path(args.output)

with open(input_path, 'r') as input_file, open(output_path, 'w+') as output_file:
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    row_count = 0
    for row in csv_reader:
        # Remove swl survey questions
        current_row = [row[i] for i in range(len(row)) if i not in args.swl_questions]
        swl_row_data = sum(RESPONE_SCORE_MAP[row[i].strip().lower()] for i in args.swl_questions) if row_count != 0 else args.swl_header_name
        current_row.insert(min(args.swl_questions), swl_row_data)
        csv_writer.writerow(current_row)

        row_count += 1
     