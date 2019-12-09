'''
Generate a weighted graph based on two-dimensional data.
'''

import csv
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
from utils import init_logger

logger = init_logger()

parser = argparse.ArgumentParser(description='Generate a weighted graph based on two-dimensional data.')
parser.add_argument('input', type=str, help='The path to the raw data CSV file.')
parser.add_argument('-x', '--independent-variable', dest='x_column', type=int, help='The (zero-based) index of the column representing the independent variable (x).', default=0)
parser.add_argument('-y', '--dependent-variable', dest='y_column', type=int, help='The (zero-based) index of the column representing the dependent variable (y).', default=1)
parser.add_argument('--header-row', type=int, help='The (zero-based) index of the header row.', default=0)
parser.add_argument('--matplotlib-style', type=str, help='The matplotlib graph style.', default='default')
parser.add_argument('--export', dest='export', help='Enable export to file.', action='store_true')
parser.add_argument('--no-export', dest='export', help='Disable export to file.', action='store_false')
parser.add_argument('--export-output', type=str, help='The path to the exported file.', default=None)
parser.add_argument('--export-dpi', type=int, help='The DPI of the exported file.', default=400)
parser.add_argument('--export-format', type=str, help='The format of the exported file.', default='png')
parser.add_argument('--no-preview', dest='preview', help='Disable the graph preview window.', action='store_false')
parser.add_argument('--show-frequencies', dest='show_frequencies', help='Display the frequency of each point.', action='store_true')
parser.add_argument('--scale-point', dest='scale_point', help='Scale the point sizes with the frequency.', action='store_true')
parser.add_argument('--base-radius-multiplier', type=float, help='The radius multiplier of a point.', default=100)
parser.set_defaults(export=False, preview=True, show_frequencies=False, scale_point=False)
args = parser.parse_args()

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
    '''
    A frequency based weight predictor function.
    '''

    return frequency[(x, y)]

def compute_weights(X, y, weight_func=None):
    '''
    Compute the weights from a set of data and a weight function.
    '''

    weights = None
    if weight_func is not None:
        weights = np.array([weight_func(X[i], y[i]) for i in range(len(X))])

    return weights

def generate_polynomial_trendline(X, y, weight_func=None, degree=1):
    '''
    Generate a polynomial fit trendline.
    '''

    weights = compute_weights(X, y, weight_func)
    trendline = np.poly1d(np.polyfit(X, y, degree, w=weights))
    r_squared = coefficient_of_determination(X, y, trendline, weight_func)
    return trendline, r_squared

def coefficient_of_determination(X, y, trendline, weight_func=None):
    '''
    Calculate the coefficient of determination (R-squared value).
    '''

    if weight_func is not None:
        weights = compute_weights(X, y, weight_func)
    else:
        # All points are equally weighted
        weights = [1] * len(X)

    y_regression = trendline(X)  

    mean = np.sum(y * weights) / np.sum(weights)

    # Calculate the total sum of squares (tss) and
    # the residual sum of squares.
    tss = np.sum(weights * (y - mean)**2)
    rss = np.sum(weights * (y_regression - y)**2)

    return 1 - rss / tss

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

    # Calculate weighting of points as a function of frequency
    frequency_weight = np.array(list(frequency.values()))

    # Split points into their independent components
    unique_points = list(frequency.keys())
    unique_x, unique_y = map(np.array, list(zip(*unique_points)))

    # Scatter plot
    sizes = None
    if args.scale_point:
        sizes = np.sqrt(frequency_weight) * args.base_radius_multiplier

    scatter = plt.scatter(unique_x, unique_y, c=frequency_weight, s=sizes, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar(scatter)

    # Sort the Xs so that matplotlib can properly display them.
    sorted_unique_x = sorted(unique_x)

    # Regression line
    unweighted_trendline, unweighted_rsquared = generate_polynomial_trendline(unique_x, unique_y)
    plt.plot(sorted_unique_x, unweighted_trendline(sorted_unique_x), linestyle='--', label='Unweighted regression')

    frequency_weighted_trendline, frequency_weighted_rsquared = generate_polynomial_trendline(unique_x, unique_y, frequency_based_weight)
    plt.plot(sorted_unique_x, frequency_weighted_trendline(sorted_unique_x), label='Frequency weighted regression')

    variance_weighted_trendline, variance_weighted_rsquared = generate_polynomial_trendline(unique_x, unique_y, variance_based_weight)
    plt.plot(sorted_unique_x, variance_weighted_trendline(sorted_unique_x), label='Variance weighted regression')

    print('Unweighted R-squared (linear):', round(unweighted_rsquared, 3))
    print('Frequency weighted R-squared (linear):', round(frequency_weighted_rsquared, 3))
    print('Variance weighted R-squared (linear):', round(variance_weighted_rsquared, 3))

    if args.show_frequencies:
        for point in unique_points:
            x, y = point
            plt.text(x + 0.3, y + 0.3, round(frequency_based_weight(x, y), 1), fontsize=9)

    # Configure plot settings
    matplotlib.style.use(args.matplotlib_style)
    plt.title('{} vs. {}'.format(y_title, x_title))
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc='upper right')

    if args.preview:
        if not args.export:
            plt.show()
        else:
            logger.warning('Graph preview was enabled but it could not be displayed since export was also enabled.' +
                ' Previewing and exporting cannot both be enabled.')

    if args.export:
        output_format = args.export_format[1:] if args.export_format.startswith('.') else args.export_format
        if args.export_output is None:
            output_extension = output_format
            if output_extension == 'latex':
                output_extension = 'tex'
            
            output_path = input_path.with_suffix('.export.' + output_extension)
        else:
            output_path = Path(args.export_output)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == 'latex':
            import tikzplotlib
            tikzplotlib.save(output_path)
        else:
            plt.savefig(output_path, dpi=args.export_dpi)