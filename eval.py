#!/usr/bin/env python

###################################################################################
# HYBRID EXPERIMENTS
###################################################################################

from __future__ import print_function
import os
import subprocess
import argparse
import pprint
import numpy
import sys
import re
import logging
import fnmatch
import string
import argparse
import pylab
import datetime
import math
import time
import fileinput
from lxml import etree

import numpy as np
import matplotlib.pyplot as plot

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LinearLocator
from pprint import pprint, pformat
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from operator import add
import matplotlib.font_manager as font_manager

import csv
import brewer2mpl
import matplotlib

from options import *
from functools import wraps

###################################################################################
# LOGGING CONFIGURATION
###################################################################################

LOG = logging.getLogger(__name__)
LOG_handler = logging.StreamHandler()
LOG_formatter = logging.Formatter(
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S'
)
LOG_handler.setFormatter(LOG_formatter)
LOG.addHandler(LOG_handler)
LOG.setLevel(logging.INFO)

###################################################################################
# OUTPUT CONFIGURATION
###################################################################################

BASE_DIR = os.path.dirname(__file__)
OPT_FONT_NAME = 'Helvetica'
OPT_GRAPH_HEIGHT = 300
OPT_GRAPH_WIDTH = 400

# Make a list by cycling through the colors you care about
# to match the length of your data.

NUM_COLORS = 5
COLOR_MAP = ( '#F58A87', '#80CA86', '#9EC9E9', '#CFAB86', '#D89761' )


#COLOR_MAP = ('#F15854', '#9C9F84', '#F7DCB4', '#991809', '#5C755E', '#A97D5D')
OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = ([ "////", "////", "o", "o", "\\\\" , "\\\\" , "//////", "//////", ".", "." , "\\\\\\" , "\\\\\\" ])

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = COLOR_MAP
OPT_LINE_WIDTH = 6.0
OPT_MARKER_SIZE = 10.0
DATA_LABELS = []


OPT_STACK_COLORS = ('#AFAFAF', '#F15854', '#5DA5DA', '#60BD68',  '#B276B2', '#DECF3F', '#F17CB0', '#B2912F', '#FAA43A')
OPT_LINE_STYLES= ('-', ':', '--', '-.')

# SET FONT

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TINY_FONT_SIZE = 8
LEGEND_FONT_SIZE = 16

SMALL_LABEL_FONT_SIZE = 10
SMALL_LEGEND_FONT_SIZE = 10

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

# SET TYPE1 FONTS
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{euler}']

LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE, weight='bold')
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)
TINY_FP = FontProperties(style='normal', size=TINY_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE, weight='bold')

SMALL_LABEL_FP = FontProperties(style='normal', size=SMALL_LABEL_FONT_SIZE, weight='bold')
SMALL_LEGEND_FP = FontProperties(style='normal', size=SMALL_LEGEND_FONT_SIZE, weight='bold')

YAXIS_TICKS = 3
YAXIS_ROUND = 1000.0

###################################################################################
# CONFIGURATION
###################################################################################

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

PELOTON_BUILD_DIR = BASE_DIR + "/../peloton/build"
LOGGING = PELOTON_BUILD_DIR + "/tests/logging_test"

OUTPUT_FILE = "outputfile.summary"

WORKLOAD_DIR = BASE_DIR + "/results/workload/"

LAYOUTS = ("row", "column", "hybrid")
OPERATORS = ("direct", "aggregate")
REORG_LAYOUTS = ("row", "hybrid")

SCALE_FACTOR = 100.0

WORKLOAD_COUNT = (10, 20, 50, 100)

TRANSACTION_COUNT = 3

WORKLOAD_EXPERIMENT = 1

###################################################################################
# UTILS
###################################################################################

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def loadDataFile(n_rows, n_cols, path):
    file = open(path, "r")
    reader = csv.reader(file)

    data = [[0 for x in xrange(n_cols)] for y in xrange(n_rows)]

    row_num = 0
    for row in reader:
        column_num = 0
        for col in row:
            data[row_num][column_num] = float(col)
            column_num += 1
        row_num += 1

    return data

def next_power_of_10(n):
    return (10 ** math.ceil(math.log(n, 10)))

def get_upper_bound(n):
    return (math.ceil(n / YAXIS_ROUND) * YAXIS_ROUND)

# # MAKE GRID
def makeGrid(ax):
    axes = ax.get_axes()
    axes.yaxis.grid(True, color=OPT_GRID_COLOR)
    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(AXIS_LINEWIDTH)
    ax.set_axisbelow(True)

# # SAVE GRAPH
def saveGraph(fig, output, width, height):
    size = fig.get_size_inches()
    dpi = fig.get_dpi()
    LOG.debug("Current Size Inches: %s, DPI: %d" % (str(size), dpi))

    new_size = (width / float(dpi), height / float(dpi))
    fig.set_size_inches(new_size)
    new_size = fig.get_size_inches()
    new_dpi = fig.get_dpi()
    LOG.debug("New Size Inches: %s, DPI: %d" % (str(new_size), new_dpi))

    pp = PdfPages(output)
    fig.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    LOG.info("OUTPUT: %s", output)

###################################################################################
# PLOT
###################################################################################

def create_workload_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    x_values = WORKLOAD
    N = len(x_values)
    x_labels = WORKLOAD

    layouts = ["NSM", "DSM", "FSM"]

    ind = np.arange(N)
    margin = 0.15
    width = ((1.0 - 2 * margin) / N)
    bars = [None] * len(layouts) * N

    print(datasets)

    for group in xrange(len(datasets)):
        # GROUP
        latencies = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    latencies.append(datasets[group][line][col])

        LOG.info("%s group_data = %s ", layouts, str(latencies))

        bars[group] = ax1.bar(ind + margin + (group * width), latencies, width,
                              color=OPT_COLORS[group],
                              hatch=OPT_PATTERNS[group*2],
                              linewidth=BAR_LINEWIDTH)


    # GRID
    axes = ax1.get_axes()
    #axes.set_ylim(0.01, 1000000)
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_ylim([YAXIS_MIN, YAXIS_MAX])

    # X-AXIS
    ax1.set_xlabel("Fraction of Attributes Projected", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    ax1.set_xticks(ind + 0.5)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

###################################################################################
# PLOT HELPERS
###################################################################################

# WORKLOAD -- PLOT
def workload_plot():

    column_count_type = 0
    for column_count in COLUMN_COUNTS:
        column_count_type = column_count_type + 1

        for write_ratio in WRITE_RATIOS:

            for operator in OPERATORS:
                print(operator)
                datasets = []

                for layout in LAYOUTS:
                    data_file = WORKLOAD_DIR + "/" + layout + "/" + operator + "/" + str(column_count) + "/" + str(write_ratio) + "/" + "projectivity.csv"

                    dataset = loadDataFile(4, 2, data_file)
                    datasets.append(dataset)

                fig = create_projectivity_bar_chart(datasets)

                if write_ratio == 0:
                    write_mix = "rd"
                else:
                    write_mix = "rw"

                if column_count_type == 1:
                    table_type = "narrow"
                else:
                    table_type = "wide"

                fileName = "workload-" + operator + "-" + table_type + "-" + write_mix + ".pdf"

                saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

###################################################################################
# EVAL HELPERS
###################################################################################

# CLEAN UP RESULT DIR
def clean_up_dir(result_directory):

    subprocess.call(['rm', '-rf', result_directory])
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

# RUN EXPERIMENT
def run_experiment(program,
                   experiment_type):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    subprocess.call([program,
                     "-e", str(experiment_type),
                     "-k", str(SCALE_FACTOR),
                     "-t", str(TRANSACTION_COUNT)])


# COLLECT STATS
def collect_stats(result_dir,
                  result_file_name,
                  category):

    fp = open(OUTPUT_FILE)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        data = line.split()

        # Collect info
        if category != DISTRIBUTION_EXPERIMENT:
            layout = data[0]
            operator = data[1]
            selectivity = data[2]
            projectivity = data[3]
            column_count = data[4]
            write_ratio = data[5]
            subset_experiment_type = data[6]
            access_num_group = data[7]
            subset_ratio = data[8]
            tuples_per_tg = data[9]
            txn_itr = data[10]
            theta = data[11]
            split_point = data[12]
            sample_weight = data[13]
            scale_factor = data[14]
            stat = data[15]

            if(layout == "0"):
                layout = "row"
            elif(layout == "1"):
                layout = "column"
            elif(layout == "2"):
                layout = "hybrid"

            if(operator == "1"):
                operator = "direct"
            elif(operator == "2"):
                operator = "aggregate"
            elif(operator == "3"):
                operator = "arithmetic"
        # Dist experiment
        else:
            query_itr = data[0]
            tile_group_type = data[1]
            tile_group_count = data[2]

        # MAKE RESULTS FILE DIR
        if category == WORKLOAD_EXPERIMENT:
            result_directory = result_dir + "/" + layout + "/" + operator + "/" + column_count + "/" + write_ratio

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")

        # WRITE OUT STATS
        if category == WORKLOAD_EXPERIMENT:
            result_file.write(str(projectivity) + " , " + str(stat) + "\n")

        result_file.close()

###################################################################################
# EVAL
###################################################################################

# WORKLOAD -- EVAL
def workload_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(WORKLOAD_DIR)

    # RUN EXPERIMENT
    run_experiment(LOGGING, WORKLOAD_EXPERIMENT)

    # COLLECT STATS
    collect_stats(WORKLOAD_DIR, "workload.csv", WORKLOAD_EXPERIMENT)

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Tilegroup Experiments')

    parser.add_argument("-w", "--workload", help='eval workload', action='store_true')

    parser.add_argument("-a", "--workload_plot", help='plot workload', action='store_true')

    args = parser.parse_args()

    if args.workload:
        workload_eval()

    if args.workload_plot:
        workload_plot()

