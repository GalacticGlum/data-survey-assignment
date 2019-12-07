'''
Generate a weighted graph based on two-dimensional data.
'''

import csv
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description='Generate a weighted graph based on two-dimensional data.')
parser.add_argument('input', type=str, help='The path to the raw data CSV file.')
parser.add_argument('-x', '--independent-variable', dest='x_column', type=int, help='The (zero-based) index of the column representing the independent variable (x).', default=0)
parser.add_argument('-y', '--dependent-variable', dest='y_column', type=int, help='The (zero-based) index of the column representing the dependent variable (y).', default=1)
parser.add_argument('--header-row', type=int, help='The (zero-based) index of the header row.', default=0)
parser.add_argument('--matplotlib-style', type=str, help='The matplotlib graph style.', default='default')
args = parser.parse_args()

# Initialize matplotlib style
matplotlib.style.use(args.matplotlib_style)

input_path = Path(args.input)
if not (input_path.is_file() or input_path.exists()):
    logger.error('The specified input is not a file or does not exist!')
    exit(1)

with open(input_path, 'r') as input_file:
    csv_reader = csv.reader(input_file)
    
    x_title = 'X'
    y_title = 'Y'

    data = {}
    frequency = {}
    
    row_count = 0
    for row in csv_reader:
        is_header_row = row_count == args.header_row
        xi, yi = map(str if is_header_row else float, (row[args.x_column], row[args.y_column]))
        if is_header_row:
            x_title, y_title = xi, yi
        else:
            if xi not in data:
                data[xi] = []

            data[xi].append(yi)
            point = (xi, yi)
            if point not in frequency:
                frequency[point] = 0

            frequency[point] += 1

        row_count += 1

    total_frequency = sum(frequency.values())

    # Get unique points
    unique_points = list(frequency.keys())
    unique_x, unique_y = [point[0] for point in unique_points], [point[1] for point in unique_points]

    # Calculate weighting of points as a function of frequency
    relative_frequencies = [frequency[point] / total_frequency for point in unique_points]
    max_relative_frequencies = max(relative_frequencies)

    color_map = plt.cm.get_cmap('Blues')
    frequency_weight = [r / max_relative_frequencies for r in relative_frequencies]

    # scatter plot
    scatter = plt.scatter(unique_x, unique_y, c=frequency_weight, cmap=color_map)
    plt.colorbar(scatter)

    # regression line
    unweighted_trendline = np.poly1d(np.polyfit(unique_x, unique_y, 1))
    plt.plot(unique_x, unweighted_trendline(unique_x), linestyle='--', label='Unweighted regression')

    weighted_trendline = np.poly1d(np.polyfit(unique_x, unique_y, 1, w=relative_frequencies))
    plt.plot(unique_x, weighted_trendline(unique_x), color='red', label='Weighted regression')

    plt.title('{} vs. {}'.format(y_title, x_title))
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc='upper right')
    plt.show()