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

def k_nearest_neighbours(source, points, k=3):
    '''
    Naive implementation of the k-nearest-neighbours search algorithm.
    '''

    n = len(points)
    distances = []
    for i in range(n):
        if points[i] == source: continue
        distance = (points[i][0] - source[0])**2 + (points[i][1] - source[1])**2
        distances.append((distance, i))
    
    ranking = [points[i[1]] for i in sorted(distances)[:k]]
    return ranking

def variance_based_weight(x, y):
    '''
    A variance-based weight predictor function.
    '''

    neighbours = k_nearest_neighbours((x, y), unique_points)
    variance = np.var([p[1] for p in filter(lambda x: x in neighbours, points)])
    return 1 / variance if variance > 0 else 0

def frequency_based_weight(x, y):
    return relative_frequencies[(x, y)]

def generate_trendline(X, y, weight_func=None, degree=1):
    '''
    Generate a polynomial fit trendline.
    '''

    weights = None
    if weight_func is not None:
        weights = [weight_func(X[i], y[i]) for i in range(len(X))]
    
    return np.poly1d(np.polyfit(X, y, degree, w=weights))

with open(input_path, 'r') as input_file:
    csv_reader = csv.reader(input_file)
    
    x_title = 'X'
    y_title = 'Y'

    frequency = {}
    points = []

    row_count = 0
    for row in csv_reader:
        is_header_row = row_count == args.header_row
        xi, yi = map(str if is_header_row else float, (row[args.x_column], row[args.y_column]))
        if is_header_row:
            x_title, y_title = xi, yi
        else:
            point = (xi, yi)
            points.append(point)

            if point not in frequency:
                frequency[point] = 0

            frequency[point] += 1

        row_count += 1

    total_frequency = sum(frequency.values())

    # Get unique points
    unique_points = list(frequency.keys())
    unique_x, unique_y = [point[0] for point in unique_points], [point[1] for point in unique_points]

    # Calculate weighting of points as a function of frequency
    relative_frequencies = {point: frequency[point] / total_frequency for point in unique_points}
    max_relative_frequencies = max(relative_frequencies.values())

    color_map = plt.cm.get_cmap('Blues')
    frequency_weight = [relative_frequencies[key] / max_relative_frequencies for key in relative_frequencies]

    # scatter plot
    scatter = plt.scatter(unique_x, unique_y, c=frequency_weight, cmap=color_map)
    plt.colorbar(scatter)

    # regression line
    sorted_unique_x = sorted(unique_x)

    unweighted_trendline = generate_trendline(unique_x, unique_y)
    plt.plot(sorted_unique_x, unweighted_trendline(sorted_unique_x), linestyle='--', label='Unweighted regression')

    weighted_trendline = generate_trendline(unique_x, unique_y, frequency_based_weight)
    plt.plot(sorted_unique_x, weighted_trendline(sorted_unique_x), color='red', label='Weighted regression')

    plt.title('{} vs. {}'.format(y_title, x_title))
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc='upper right')
    plt.show()