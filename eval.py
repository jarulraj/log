#!/usr/bin/env python

###################################################################################
# HYBRID EXPERIMENTS
###################################################################################

from __future__ import print_function

import csv
import logging
import math
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator
import os
import pprint
import pylab
import subprocess
import argparse

import matplotlib.pyplot as plot
import numpy as np


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

# http://colrd.com/palette/19308/
COLOR_MAP = ('#51574a', '#447c69', '#74c493',
             '#8e8c6d', '#e4bf80', '#e9d78e',
             '#e2975d', '#f19670', '#e16552',
             '#c94a53', '#be5168', '#a34974',
             '#993767', '#65387d', '#4e2472',
             '#9163b6', '#e279a3', '#e0598b',
             '#7c9fb0', '#5698c4', '#9abf88')

OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = ([ "////", "////", "o", "o", "\\\\" , "\\\\" , "//////", "//////", ".", "." , "\\\\\\" , "\\\\\\" ])

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = COLOR_MAP
OPT_LINE_WIDTH = 3.0
OPT_MARKER_SIZE = 6.0
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
LOGGING = PELOTON_BUILD_DIR + "/src/.libs/logger"

SDV_DIR = "/data/devel/sdv-tools/sdv-release"
SDV_SCRIPT = SDV_DIR + "/ivt_pm_sdv.sh"
NVM_LATENCIES = ("160", "320")
DEFAULT_NVM_LATENCY = NVM_LATENCIES[0]
ENABLE_SDV = False

OUTPUT_FILE = "outputfile.summary"

# Refer LoggingType in common/types.h
LOGGING_TYPES = (0, 1, 2, 3, 4)

LOGGING_NAMES = ("disabled", 
                 "nvm_wal", "nvm_wbl", 
                 "hdd_wal", "hdd_wbl")

SCALE_FACTOR = 1
DATABASE_FILE_SIZE = 4096  # DATABASE FILE SIZE (MB)

TRANSACTION_COUNT = 10

CLIENT_COUNTS = (1, 2, 4, 8)

YCSB_UPDATE_RATIOS = (0, 0.5)
YCSB_UPDATE_NAMES = ("read_only", "balanced")

YCSB_WORKLOAD_DIR = BASE_DIR + "/results/workload/ycsb/"
YCSB_WORKLOAD_EXPERIMENT = 1
YCSB_WORKLOAD_CSV = "ycsb_workload.csv"

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

# Figure out ycsb update name
def getYCSBUpdateName(ycsb_update_ratio):

    ycsb_update_ratio_type_offset = YCSB_UPDATE_RATIOS.index(float(ycsb_update_ratio))
    ycsb_update_name = YCSB_UPDATE_NAMES[ycsb_update_ratio_type_offset]

    return ycsb_update_name

# Figure out logging name
def getLoggingName(logging_type):
    
    logging_type_offset = LOGGING_TYPES.index(int(logging_type))
    logging_name = LOGGING_NAMES[logging_type_offset]

    return logging_name

###################################################################################
# PLOT
###################################################################################

def create_legend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(12, 3.0))

    N = len(LOGGING_NAMES);
    ind = np.arange(1)
    margin = 0.10
    width = ((1.0 - 2 * margin) / N) * 2
    data = [1]

    bars = [None] * (len(LOGGING_NAMES) + 1) * 2

    idx = 0
    for group in xrange(len(LOGGING_NAMES)):
        bars[idx] = ax1.bar(ind + margin + ((idx + 1) * width), data, width,
                            color=OPT_COLORS[idx],
                            linewidth=BAR_LINEWIDTH)
        idx = idx + 1

    # LEGEND
    figlegend.legend(bars, LOGGING_NAMES, prop=LEGEND_FP,
                     loc=1, ncol=5,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1.5, handlelength=4)

    figlegend.savefig('legend.pdf')

def create_ycsb_workload_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = np.arange(len(CLIENT_COUNTS))
    N = len(x_values)
    x_labels = [str(i) for i in CLIENT_COUNTS]

    M = len(LOGGING_NAMES)
    ind = np.arange(N)
    margin = 0.15
    width = ((1.0 - 2 * margin) / M)
    bars = [None] * M * N

    for group in xrange(len(datasets)):
        # GROUP
        group_data = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    group_data.append(datasets[group][line][col])

        LOG.info("group_data = %s", str(group_data))

        bars[group] = ax1.bar(ind + margin + (group * width), group_data, width,
                              color=OPT_COLORS[group],
                              linewidth=BAR_LINEWIDTH)


    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Throughput", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xlabel("Number of Clients", fontproperties=LABEL_FP)
    ax1.set_xticks(ind + margin + (group * width)/2.0 )
    ax1.set_xticklabels(x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)


###################################################################################
# PLOT HELPERS
###################################################################################

# WORKLOAD -- PLOT
def ycsb_workload_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for logging_type in LOGGING_TYPES:

            # figure out logging name and ycsb update name
            logging_name = getLoggingName(logging_type)
    
            data_file = YCSB_WORKLOAD_DIR + "/" + ycsb_update_name + "/" + logging_name + "/" + YCSB_WORKLOAD_CSV
    
            dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
            datasets.append(dataset)

        fig = create_ycsb_workload_bar_chart(datasets)
    
        fileName = "ycsb_" + ycsb_update_name + ".pdf"
    
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
                   logging_type,
                   client_count,
                   ycsb_update_ratio):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    subprocess.call([program,
                     "-e", str(experiment_type),
                     "-l", str(logging_type),
                     "-k", str(SCALE_FACTOR),
                     "-f", str(DATABASE_FILE_SIZE),
                     "-t", str(TRANSACTION_COUNT),
                     "-b", str(client_count),
                     "-u", str(ycsb_update_ratio)])


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
        logging_type = data[0]
        ycsb_update_ratio = data[1]
        scale_factor = data[2]
        backend_count = data[3]

        stat = data[4]

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)
        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        # MAKE RESULTS FILE DIR
        if category == YCSB_WORKLOAD_EXPERIMENT:
            result_directory = result_dir + "/" + ycsb_update_name + "/" + logging_name

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")

        # WRITE OUT STATS
        if category == YCSB_WORKLOAD_EXPERIMENT:
            result_file.write(str(backend_count) + " , " + str(stat) + "\n")

        result_file.close()

###################################################################################
# EVAL
###################################################################################

def set_nvm_latency(nvm_latency):
    if ENABLE_SDV :
        FNULL = open(os.devnull, 'w')
        cwd = os.getcwd()
        os.chdir(SDV_DIR)
        subprocess.call(['sudo', SDV_SCRIPT, '--enable', '--pm-latency', str(nvm_latency)], stdout=FNULL)
        os.chdir(cwd)
        FNULL.close()

def reset_nvm_latency():
    if ENABLE_SDV :
        FNULL = open(os.devnull, 'w')
        cwd = os.getcwd()
        os.chdir(SDV_DIR)
        subprocess.call(['sudo', SDV_SCRIPT, '--enable', '--pm-latency', str(DEFAULT_NVM_LATENCY)], stdout=FNULL)
        os.chdir(cwd)
        FNULL.close()

# WORKLOAD -- EVAL
def ycsb_workload_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(YCSB_WORKLOAD_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for logging_type in LOGGING_TYPES:
            for client_count in CLIENT_COUNTS:
                
                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               YCSB_WORKLOAD_EXPERIMENT,
                               logging_type,
                               client_count,
                               ycsb_update_ratio)

                # COLLECT STATS
                collect_stats(YCSB_WORKLOAD_DIR, YCSB_WORKLOAD_CSV, YCSB_WORKLOAD_EXPERIMENT)


###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Write Behind Logging Experiments')

    parser.add_argument("-x", "--enable-sdv", help='enable sdv', action='store_true')
    parser.add_argument("-w", "--ycsb_workload", help='eval ycsb_workload', action='store_true')

    parser.add_argument("-a", "--ycsb_workload_plot", help='plot ycsb_workload', action='store_true')

    args = parser.parse_args()

    if args.enable_sdv:
        ENABLE_SDV = os.path.exists(SDV_DIR)
        print("ENABLE_SDV : " + str(ENABLE_SDV))

    ## EVAL

    if args.ycsb_workload:
        ycsb_workload_eval()

    ## PLOT

    if args.ycsb_workload_plot:
        ycsb_workload_plot()

    #create_legend()
