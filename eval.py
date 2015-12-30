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
RECOVERY_DIR = BASE_DIR + "/results/recovery/"
STORAGE_DIR = BASE_DIR + "/results/storage/"

WORKLOAD_COUNT = (10, 20, 50, 100)
COLUMN_COUNTS = (5, 10, 20, 50)
TUPLE_COUNTS = (100, 1000, 10000)

LOGGING_TYPES = (0, 1, 2)
LOGGING_NAMES = ("NONE", "WAL", "WBL")

# Skip no logging
LOGGING_TYPES_SUBSET = (1, 2)   
LOGGING_NAMES_SUBSET = ("WAL", "WBL")

DEFAULT_TUPLE_COUNT = 10000
DEFAULT_COLUMN_COUNT = 20

WORKLOAD_EXPERIMENT = 1
RECOVERY_EXPERIMENT = 2
STORAGE_EXPERIMENT = 3

SDV_DIR = "/data/devel/sdv-tools/sdv-release"
SDV_SCRIPT = SDV_DIR + "/ivt_pm_sdv.sh"    
NVM_LATENCIES = ("160", "320", "480")
DEFAULT_NVM_LATENCY = NVM_LATENCIES[0]
ENABLE_SDV = False

PMEM_FILE_DIR = "/mnt/pmfs/peloton/"
PMEM_FILE_SIZE = 512

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

    # X-AXIS
    x_values = np.arange(len(COLUMN_COUNTS))
    N = len(x_values)
    x_labels = [str(i) for i in COLUMN_COUNTS]

    ind = np.arange(N)
    idx = 0

    # GROUP
    for group_index, group in enumerate(LOGGING_TYPES):
        group_data = []

        # LINE
        for line_index, line in enumerate(x_values):
            group_data.append(datasets[group_index][line_index][1])

        LOG.info("%s group_data = %s ", group, str(group_data))

        ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xlabel("Tuple width", fontproperties=LABEL_FP)
    plot.xticks(x_values, x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_recovery_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = np.arange(len(TUPLE_COUNTS))
    N = len(x_values)
    x_labels = [str(i) for i in TUPLE_COUNTS]

    ind = np.arange(N)
    idx = 1

    # GROUP
    for group_index, group in enumerate(LOGGING_TYPES_SUBSET):
        group_data = []

        # LINE
        for line_index, line in enumerate(x_values):
            group_data.append(datasets[group_index][line_index][1])

        LOG.info("%s group_data = %s ", group, str(group_data))

        ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xlabel("Number of transactions", fontproperties=LABEL_FP)
    plot.xticks(x_values, x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_storage_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = np.arange(len(TUPLE_COUNTS))
    N = len(x_values)
    x_labels = [str(i) for i in TUPLE_COUNTS]

    ind = np.arange(N)
    idx = 1

    # GROUP
    for group_index, group in enumerate(LOGGING_TYPES_SUBSET):
        group_data = []

        # LINE
        for line_index, line in enumerate(x_values):
            group_data.append(datasets[group_index][line_index][1])

        LOG.info("%s group_data = %s ", group, str(group_data))

        ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Log Storage Footprint (B)", fontproperties=LABEL_FP)
    ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xlabel("Number of transactions", fontproperties=LABEL_FP)
    plot.xticks(x_values, x_labels)

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

    for nvm_latency in NVM_LATENCIES:
        datasets = []

        for logging_name in LOGGING_NAMES:

            data_file = WORKLOAD_DIR + "/" + logging_name + "/" + str(nvm_latency) + "/" + "workload.csv"
    
            dataset = loadDataFile(len(COLUMN_COUNTS), 2, data_file)
            datasets.append(dataset)

        fig = create_workload_bar_chart(datasets)
    
        fileName = "workload-" + str(nvm_latency) + ".pdf"
    
        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# RECOVERY -- PLOT
def recovery_plot():

    for nvm_latency in NVM_LATENCIES:

        for column_count in COLUMN_COUNTS:
            datasets = []
            
            for logging_name in LOGGING_NAMES_SUBSET:
    
                data_file = RECOVERY_DIR + "/" + logging_name + "/" + str(nvm_latency) + "/" + str(column_count) + "/" + "recovery.csv"
    
                dataset = loadDataFile(len(TUPLE_COUNTS), 2, data_file)
                datasets.append(dataset)
    
            fig = create_recovery_bar_chart(datasets)
    
            fileName = "recovery-" + str(nvm_latency) + "-" + str(column_count) + ".pdf"
    
            saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# STORAGE -- PLOT
def storage_plot():

    for column_count in COLUMN_COUNTS:
        datasets = []

        for logging_name in LOGGING_NAMES_SUBSET:
    
            data_file = STORAGE_DIR + "/" + logging_name + "/" + str(column_count) + "/" + "storage.csv"
    
            dataset = loadDataFile(len(TUPLE_COUNTS), 2, data_file)
            datasets.append(dataset)
    
        fig = create_storage_bar_chart(datasets)
    
        fileName = "storage-" + str(column_count) + ".pdf"
    
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
                   experiment_type,
                   column_count,
                   tuple_count,
                   logging_type):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    subprocess.call([program,
                     "-e", str(experiment_type),
                     "-t", str(tuple_count),
                     "-l", str(logging_type),
                     "-z", str(column_count),
                     "-d", str(PMEM_FILE_DIR),
                     "-f", str(PMEM_FILE_SIZE)])


# COLLECT STATS
def collect_stats(result_dir,
                  result_file_name,
                  category,
                  nvm_latency):

    fp = open(OUTPUT_FILE)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        data = line.split()

        # Collect info
        logging_type = data[0]
        column_count = data[1]
        tuple_count = data[2]
        backend_count = data[3]
        
        stat = data[4]

        if(logging_type == "0"):
            logging_name = LOGGING_NAMES[0]
        elif(logging_type == "1"):
            logging_name = LOGGING_NAMES[1]
        elif(logging_type == "2"):
            logging_name = LOGGING_NAMES[2]

        # MAKE RESULTS FILE DIR
        if category == WORKLOAD_EXPERIMENT:
            result_directory = result_dir + "/" + logging_name + "/" + str(nvm_latency)
        elif category == RECOVERY_EXPERIMENT:
            result_directory = result_dir + "/" + logging_name + "/" + str(nvm_latency) + "/" + str(column_count)            
        elif category == STORAGE_EXPERIMENT:
            result_directory = result_dir + "/" + logging_name + "/" + str(column_count)

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")

        # WRITE OUT STATS
        if category == WORKLOAD_EXPERIMENT:
            result_file.write(str(column_count) + " , " + str(stat) + "\n")
        elif category == RECOVERY_EXPERIMENT or category == STORAGE_EXPERIMENT:
            result_file.write(str(tuple_count) + " , " + str(stat) + "\n")            
            
        result_file.close()

###################################################################################
# EVAL
###################################################################################

def set_nvm_latency(nvm_latency):
    if ENABLE_SDV :
        cwd = os.getcwd()
        os.chdir(SDV_DIR)
        subprocess.call(['sudo', SDV_SCRIPT, '--enable', '--pm-latency', str(nvm_latency)], stdout=log_file)
        os.chdir(cwd)

def reset_nvm_latency():
    if ENABLE_SDV :
        cwd = os.getcwd()
        os.chdir(SDV_DIR)
        subprocess.call(['sudo', SDV_SCRIPT, '--enable', '--pm-latency', str(DEFAULT_NVM_LATENCY)], stdout=log_file)
        os.chdir(cwd)

# WORKLOAD -- EVAL
def workload_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(WORKLOAD_DIR)

    tuple_count = DEFAULT_TUPLE_COUNT

    for nvm_latency in NVM_LATENCIES:
        # SET NVM LATENCY
        set_nvm_latency(nvm_latency)
        
        for logging_type in LOGGING_TYPES:        
            for column_count in COLUMN_COUNTS:
    
                # RUN EXPERIMENT            
                run_experiment(LOGGING, WORKLOAD_EXPERIMENT, column_count, tuple_count, logging_type)
    
                # COLLECT STATS
                collect_stats(WORKLOAD_DIR, "workload.csv", WORKLOAD_EXPERIMENT, nvm_latency)

    # RESET NVM LATENCY
    reset_nvm_latency()


# RECOVERY -- EVAL
def recovery_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(RECOVERY_DIR)

    for nvm_latency in NVM_LATENCIES:
         # SET NVM LATENCY
        set_nvm_latency(nvm_latency)
        
        for logging_type in LOGGING_TYPES_SUBSET:        
            for tuple_count in TUPLE_COUNTS:
                for column_count in COLUMN_COUNTS:
    
                    # RUN EXPERIMENT            
                    run_experiment(LOGGING, RECOVERY_EXPERIMENT, column_count, tuple_count, logging_type)
        
                    # COLLECT STATS
                    collect_stats(RECOVERY_DIR, "recovery.csv", RECOVERY_EXPERIMENT, nvm_latency)

    # RESET NVM LATENCY
    reset_nvm_latency()

# STORAGE -- EVAL
def storage_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(STORAGE_DIR)
    
    for logging_type in LOGGING_TYPES_SUBSET:        
        for tuple_count in TUPLE_COUNTS:
            for column_count in COLUMN_COUNTS:

                # RUN EXPERIMENT            
                run_experiment(LOGGING, STORAGE_EXPERIMENT, column_count, tuple_count, logging_type)
    
                # COLLECT STATS
                collect_stats(STORAGE_DIR, "storage.csv", STORAGE_EXPERIMENT, 0)
            

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Hybrid Experiments')

    parser.add_argument("-x", "--enable-sdv", help='enable sdv', action='store_true')
    parser.add_argument("-w", "--workload", help='eval workload', action='store_true')
    parser.add_argument("-r", "--recovery", help='eval recovery', action='store_true')
    parser.add_argument("-s", "--storage", help='eval storage', action='store_true')

    parser.add_argument("-a", "--workload_plot", help='plot workload', action='store_true')
    parser.add_argument("-b", "--recovery_plot", help='plot recovery', action='store_true')
    parser.add_argument("-c", "--storage_plot", help='plot recovery', action='store_true')

    args = parser.parse_args()

    if args.enable_sdv:
        ENABLE_SDV = os.path.exists(SDV_DIR)
        print("ENABLE_SDV : " + str(ENABLE_SDV))
    
    ## EVAL
    
    if args.workload:
        workload_eval()

    if args.recovery:
        recovery_eval()

    if args.storage:
        storage_eval()

    ## PLOT

    if args.workload_plot:
        workload_plot()

    if args.recovery_plot:
        recovery_plot()

    if args.storage_plot:
        storage_plot()
