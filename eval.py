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
from operator import add
from pprint import pprint, pformat

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
COLOR_MAP = ('#447c69', '#74c493', '#e279a3', '#c94a53',
             '#9163b6', '#4e2472', '#8e8c6d', '#51574a')

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

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
TINY_FONT_SIZE = 10
LEGEND_FONT_SIZE = 18

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
ENABLE_SDV = False

NVM_LATENCIES = ("160", "320")
DEFAULT_NVM_LATENCY = NVM_LATENCIES[0]
INVALID_NVM_LATENCY = 0

PCOMMIT_LATENCIES = ("10", "20")
INVALID_PCOMMIT_LATENCY = 0

OUTPUT_FILE = "outputfile.summary"

# Refer LoggingType in common/types.h
LOGGING_TYPES = (1, 2, 3, 4, 5, 6)
LOGGING_NAMES = ("nvm_wal", "ssd_wal", "hdd_wal", "nvm_wbl", "ssd_wbl", "hdd_wbl")

NVM_LOGGING_TYPES = (1, 4)
NVM_LOGGING_NAMES = ("nvm_wal", "nvm_wbl")

SCALE_FACTOR = 1
DATABASE_FILE_SIZE = 4096  # DATABASE FILE SIZE (MB)

DURATION = 1000

CLIENT_COUNTS = (1, 2, 4, 8)
DEFAULT_CLIENT_COUNT = 8

YCSB_BENCHMARK_TYPE = 1
TPCC_BENCHMARK_TYPE = 2

RECOVERY_DURATIONS = (500, 5000, 50000)

YCSB_UPDATE_RATIOS = (0, 0.1, 0.5, 0.9)
YCSB_UPDATE_NAMES = ("read-only", "read-heavy", "balanced", "write-heavy")
INVALID_UPDATE_RATIO = 0

FLUSH_MODES = ("1", "2")
DEFAULT_FLUSH_MODE = 2

ASYNCHRONOUS_MODES = ("1", "2", "3")
DEFAULT_ASYNCHRONOUS_MODE = 1

EXPERIMENT_TYPE_THROUGHPUT = 1
EXPERIMENT_TYPE_RECOVERY = 2
EXPERIMENT_TYPE_STORAGE = 3
EXPERIMENT_TYPE_LATENCY = 4

STORAGE_LOGGING_TYPES = ("WAL", "WBL")
STORAGE_LABELS = ("Table", "Index", "Log", "Checkpoint", "Other")

YCSB_THROUGHPUT_DIR = BASE_DIR + "/results/throughput/ycsb/"
YCSB_THROUGHPUT_EXPERIMENT = 1
YCSB_THROUGHPUT_CSV = "ycsb_throughput.csv"

TPCC_THROUGHPUT_DIR = BASE_DIR + "/results/throughput/tpcc/"
TPCC_THROUGHPUT_EXPERIMENT = 2
TPCC_THROUGHPUT_CSV = "tpcc_throughput.csv"

YCSB_RECOVERY_DIR = BASE_DIR + "/results/recovery/ycsb/"
YCSB_RECOVERY_EXPERIMENT = 3
YCSB_RECOVERY_CSV = "ycsb_recovery.csv"

TPCC_RECOVERY_DIR = BASE_DIR + "/results/recovery/tpcc/"
TPCC_RECOVERY_EXPERIMENT = 4
TPCC_RECOVERY_CSV = "tpcc_recovery.csv"

YCSB_STORAGE_DIR = BASE_DIR + "/results/storage/ycsb/"
YCSB_STORAGE_CSV = "ycsb_storage.csv"

TPCC_STORAGE_DIR = BASE_DIR + "/results/storage/tpcc/"
TPCC_STORAGE_CSV = "tpcc_storage.csv"

YCSB_LATENCY_DIR = BASE_DIR + "/results/latency/ycsb/"
YCSB_LATENCY_EXPERIMENT = 5
YCSB_LATENCY_CSV = "ycsb_latency.csv"

TPCC_LATENCY_DIR = BASE_DIR + "/results/latency/tpcc/"
TPCC_LATENCY_EXPERIMENT = 6
TPCC_LATENCY_CSV = "tpcc_latency.csv"

NVM_LATENCY_DIR = BASE_DIR + "/results/nvm_latency/"
NVM_LATENCY_EXPERIMENT = 7
NVM_LATENCY_CSV = "nvm_latency.csv"

PCOMMIT_LATENCY_DIR = BASE_DIR + "/results/pcommit_latency/"
PCOMMIT_LATENCY_EXPERIMENT = 8
PCOMMIT_LATENCY_CSV = "pcommit_latency.csv"

FLUSH_MODE_DIR = BASE_DIR + "/results/flush_mode/"
FLUSH_MODE_EXPERIMENT = 9
FLUSH_MODE_CSV = "flush_mode.csv"

ASYNCHRONOUS_MODE_DIR = BASE_DIR + "/results/asynchronous_mode/"
ASYNCHRONOUS_MODE_EXPERIMENT = 10
ASYNCHRONOUS_MODE_CSV = "asynchronous_mode.csv"

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

def create_legend_logging_types():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(16, 0.5))

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

    LOGGING_NAMES_UPPER_CASE = [x.upper() for x in LOGGING_NAMES]

    # LEGEND
    figlegend.legend(bars, LOGGING_NAMES_UPPER_CASE, prop=LEGEND_FP,
                     loc=1, ncol=len(LOGGING_NAMES),
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_logging_types.pdf')

def create_legend_storage():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(10, 0.5))

    num_items = 5;
    ind = np.arange(1)
    margin = 0.10
    width = (1.0 - 2 * margin) / num_items

    bars = [None] * len(STORAGE_LABELS) * 2

    for group in xrange(len(STORAGE_LABELS)):
        data = [1]
        bars[group] = ax1.bar(ind + margin + (group * width), data, width,
                              color=OPT_STACK_COLORS[group], linewidth=BAR_LINEWIDTH)

    # LEGEND
    figlegend.legend(bars, STORAGE_LABELS, prop=LABEL_FP,
                     loc=1, ncol=len(STORAGE_LABELS),
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=2, handlelength=3.5)

    figlegend.savefig('legend_storage.pdf')

def create_ycsb_throughput_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in CLIENT_COUNTS]
    N = len(x_labels)
    ind = np.arange(N)

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        group_data = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    group_data.append(datasets[group][line][col])

        LOG.info("group_data = %s", str(group_data))

        ax1.plot(ind + 0.5, group_data,
                 color=OPT_LINE_COLORS[idx],
                 linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE,
                 label=str(group))

        idx = idx + 1


    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Throughput", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Number of Clients", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_ycsb_recovery_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in RECOVERY_DURATIONS]
    N = len(x_labels)
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
    ax1.set_ylabel("Recovery Latency (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Number of Clients", fontproperties=LABEL_FP)
    ax1.set_xticks(ind + margin + (group * width)/2.0 )
    ax1.set_xticklabels(x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_ycsb_storage_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    num_items = len(STORAGE_LOGGING_TYPES);
    ind = np.arange(num_items)
    margin = 0.2
    width = 1.0

    col_offset = 0.2
    col_width = width - col_offset

    bars = [None] * len(STORAGE_LABELS) * 2
    YLIMIT = len(STORAGE_LABELS)

    datasets = map(list, map(None,*datasets))

    # TYPE
    bottom_list = [0] * len(datasets[0])
    for type in  xrange(len(datasets)):
        LOG.info("TYPE :: %s", datasets[type])

        bars[type] = ax1.bar(ind + margin + col_offset, datasets[type], col_width,
                             color=OPT_STACK_COLORS[type], linewidth=BAR_LINEWIDTH,
                             bottom = bottom_list)
        bottom_list = map(add, bottom_list, datasets[type])

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.set_ylabel("Storage (GB)", fontproperties=LABEL_FP)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    axes.set_ylim(0, YLIMIT)

    # X-AXIS
    ax1.tick_params(axis='x', which='both', top='off', bottom='off')
    ax1.set_xticks(ind + margin + 0.6)
    ax1.set_xticklabels(STORAGE_LOGGING_TYPES)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_ycsb_latency_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in CLIENT_COUNTS]
    N = len(x_labels)
    ind = np.arange(N)

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        group_data = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    group_data.append(datasets[group][line][col] * 1000) # ms

        LOG.info("group_data = %s", str(group_data))

        ax1.plot(ind + 0.5, group_data,
                 color=OPT_LINE_COLORS[idx],
                 linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE,
                 label=str(group))

        idx = idx + 1


    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Number of Clients", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_nvm_latency_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in NVM_LATENCIES]
    N = len(x_labels)
    M = len(NVM_LOGGING_NAMES)
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
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("NVM Latency", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_pcommit_latency_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in PCOMMIT_LATENCIES]
    N = len(x_labels)
    M = len(NVM_LOGGING_NAMES)
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
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("PCOMMIT Latency", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_flush_mode_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in FLUSH_MODES]
    N = len(x_labels)
    M = len(NVM_LOGGING_NAMES)
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
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Flush Mode", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_asynchronous_mode_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in ASYNCHRONOUS_MODES]
    N = len(x_labels)
    M = len(NVM_LOGGING_NAMES)
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
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Asynchronous Mode", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

###################################################################################
# PLOT HELPERS
###################################################################################

# YCSB THROUGHPUT -- PLOT
def ycsb_throughput_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for logging_type in LOGGING_TYPES:

            # figure out logging name and ycsb update name
            logging_name = getLoggingName(logging_type)

            data_file = YCSB_THROUGHPUT_DIR + "/" + ycsb_update_name + "/" + logging_name + "/" + YCSB_THROUGHPUT_CSV

            dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
            datasets.append(dataset)

        fig = create_ycsb_throughput_line_chart(datasets)

        fileName = "ycsb-" + "throughput-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# TPCC THROUGHPUT -- PLOT
def tpcc_throughput_plot():

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = TPCC_THROUGHPUT_DIR + "/" + logging_name + "/" + TPCC_THROUGHPUT_CSV

        dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_throughput_line_chart(datasets)

    fileName = "tpcc-" + "throughput" + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# YCSB RECOVERY -- PLOT
def ycsb_recovery_plot():

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = YCSB_RECOVERY_DIR + "/" + logging_name + "/" + YCSB_RECOVERY_CSV

        dataset = loadDataFile(len(RECOVERY_DURATIONS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_recovery_bar_chart(datasets)

    fileName = "ycsb-" + "recovery" + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# TPCC RECOVERY -- PLOT
def tpcc_recovery_plot():

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = TPCC_RECOVERY_DIR + "/" + logging_name + "/" + TPCC_RECOVERY_CSV

        dataset = loadDataFile(len(RECOVERY_DURATIONS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_recovery_bar_chart(datasets)

    fileName = "tpcc-" + "recovery" + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# YCSB STORAGE -- PLOT
def ycsb_storage_plot():

    data_file =  os.path.realpath(os.path.join(YCSB_STORAGE_DIR, YCSB_STORAGE_CSV))

    dataset = loadDataFile(len(STORAGE_LOGGING_TYPES), len(STORAGE_LABELS) + 1, data_file)

    fig = create_ycsb_storage_bar_chart(dataset)

    fileName = "ycsb-storage.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH/2, height=OPT_GRAPH_HEIGHT/2.0)

# TPCC STORAGE -- PLOT
def tpcc_storage_plot():

    data_file =  os.path.realpath(os.path.join(TPCC_STORAGE_DIR, TPCC_STORAGE_CSV))

    dataset = loadDataFile(len(STORAGE_LOGGING_TYPES), len(STORAGE_LABELS) + 1, data_file)

    fig = create_ycsb_storage_bar_chart(dataset)

    fileName = "tpcc-storage.pdf"

    saveGraph(fig, fileName, width=OPT_GRAPH_WIDTH/2, height=OPT_GRAPH_HEIGHT/2.0)

# YCSB LATENCY -- PLOT
def ycsb_latency_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for logging_type in LOGGING_TYPES:

            # figure out logging name and ycsb update name
            logging_name = getLoggingName(logging_type)

            data_file = YCSB_LATENCY_DIR + "/" + ycsb_update_name + "/" + logging_name + "/" + YCSB_LATENCY_CSV

            dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
            datasets.append(dataset)

        fig = create_ycsb_latency_line_chart(datasets)

        fileName = "ycsb-" + "latency-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# TPCC LATENCY -- PLOT
def tpcc_latency_plot():

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = TPCC_LATENCY_DIR + "/" + logging_name + "/" + TPCC_LATENCY_CSV

        dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_latency_line_chart(datasets)

    fileName = "tpcc-" + "latency" + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# NVM LATENCY -- PLOT
def nvm_latency_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for nvm_logging_type in NVM_LOGGING_TYPES:

            # figure out logging name and ycsb update name
            nvm_logging_name = getLoggingName(nvm_logging_type)

            data_file = NVM_LATENCY_DIR + "/" + ycsb_update_name + "/" + nvm_logging_name + "/" + NVM_LATENCY_CSV

            dataset = loadDataFile(len(NVM_LATENCIES), 2, data_file)
            datasets.append(dataset)

        fig = create_nvm_latency_bar_chart(datasets)

        fileName = "nvm-latency-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# PCOMMIT LATENCY -- PLOT
def pcommit_latency_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for nvm_logging_type in NVM_LOGGING_TYPES:

            # figure out logging name and ycsb update name
            nvm_logging_name = getLoggingName(nvm_logging_type)

            data_file = PCOMMIT_LATENCY_DIR + "/" + ycsb_update_name + "/" + nvm_logging_name + "/" + PCOMMIT_LATENCY_CSV

            dataset = loadDataFile(len(PCOMMIT_LATENCIES), 2, data_file)
            datasets.append(dataset)

        fig = create_pcommit_latency_bar_chart(datasets)

        fileName = "pcommit-latency-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# FLUSH MODE -- PLOT
def flush_mode_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for nvm_logging_type in NVM_LOGGING_TYPES:

            # figure out logging name and ycsb update name
            nvm_logging_name = getLoggingName(nvm_logging_type)

            data_file = FLUSH_MODE_DIR + "/" + ycsb_update_name + "/" + nvm_logging_name + "/" + FLUSH_MODE_CSV

            dataset = loadDataFile(len(FLUSH_MODES), 2, data_file)
            datasets.append(dataset)

        fig = create_flush_mode_bar_chart(datasets)

        fileName = "flush-mode-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

# ASYNCHRONOUS MODE -- PLOT
def asynchronous_mode_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for nvm_logging_type in NVM_LOGGING_TYPES:

            # figure out logging name and ycsb update name
            nvm_logging_name = getLoggingName(nvm_logging_type)

            data_file = ASYNCHRONOUS_MODE_DIR + "/" + ycsb_update_name + "/" + nvm_logging_name + "/" + ASYNCHRONOUS_MODE_CSV

            dataset = loadDataFile(len(ASYNCHRONOUS_MODES), 2, data_file)
            datasets.append(dataset)

        fig = create_asynchronous_mode_bar_chart(datasets)

        fileName = "asynchronous-mode-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

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
                   benchmark_type,
                   client_count,
                   duration,
                   ycsb_update_ratio,
                   flush_mode,
                   nvm_latency,
                   pcommit_latency,
                   asynchronous_mode):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    subprocess.call([program,
                     "-e", str(experiment_type),
                     "-l", str(logging_type),
                     "-k", str(SCALE_FACTOR),
                     "-f", str(DATABASE_FILE_SIZE),
                     "-d", str(duration),
                     "-b", str(client_count),
                     "-u", str(ycsb_update_ratio),
                     "-y", str(benchmark_type),
                     "-v", str(flush_mode),
                     "-n", str(nvm_latency),
                     "-p", str(pcommit_latency),
                     "-a", str(asynchronous_mode)])


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
        benchmark_type = data[0]
        logging_type = data[1]
        ycsb_update_ratio = data[2]
        scale_factor = data[3]
        backend_count = data[4]
        transaction_count = data[5]
        nvm_latency = data[6]
        pcommit_latency = data[7]
        flush_mode = data[8]
        asynchronous_mode = data[9]

        stat = data[10]

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        # MAKE RESULTS FILE DIR
        if category == YCSB_THROUGHPUT_EXPERIMENT or category == YCSB_LATENCY_EXPERIMENT:
            ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)
            result_directory = result_dir + "/" + ycsb_update_name + "/" + logging_name
        elif category == TPCC_THROUGHPUT_EXPERIMENT or category == TPCC_LATENCY_EXPERIMENT:
            result_directory = result_dir + "/" + logging_name
        elif category == YCSB_RECOVERY_EXPERIMENT or category == TPCC_RECOVERY_EXPERIMENT:
            result_directory = result_dir + "/" + logging_name
        elif category == NVM_LATENCY_EXPERIMENT or category == PCOMMIT_LATENCY_EXPERIMENT \
         or category == FLUSH_MODE_EXPERIMENT or category == ASYNCHRONOUS_MODE_EXPERIMENT :
            ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)
            result_directory = result_dir + "/" + ycsb_update_name + "/" + logging_name

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")

        # WRITE OUT STATS
        if category == YCSB_THROUGHPUT_EXPERIMENT or category == TPCC_THROUGHPUT_EXPERIMENT \
         or category == YCSB_LATENCY_EXPERIMENT or category == TPCC_LATENCY_EXPERIMENT:
            result_file.write(str(backend_count) + " , " + str(stat) + "\n")
        elif category == YCSB_RECOVERY_EXPERIMENT or category == TPCC_RECOVERY_EXPERIMENT:
            result_file.write(str(transaction_count) + " , " + str(stat) + "\n")
        elif category == NVM_LATENCY_EXPERIMENT:
            result_file.write(str(nvm_latency) + " , " + str(stat) + "\n")
        elif category == PCOMMIT_LATENCY_EXPERIMENT:
            result_file.write(str(pcommit_latency) + " , " + str(stat) + "\n")
        elif category == FLUSH_MODE_EXPERIMENT:
            result_file.write(str(flush_mode) + " , " + str(stat) + "\n")
        elif category == ASYNCHRONOUS_MODE_EXPERIMENT:
            result_file.write(str(asynchronous_mode) + " , " + str(stat) + "\n")

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

# YCSB THROUGHPUT -- EVAL
def ycsb_throughput_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(YCSB_THROUGHPUT_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for logging_type in LOGGING_TYPES:
            for client_count in CLIENT_COUNTS:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               client_count,
                               DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE)

                # COLLECT STATS
                collect_stats(YCSB_THROUGHPUT_DIR, YCSB_THROUGHPUT_CSV, YCSB_THROUGHPUT_EXPERIMENT)

# TPCC THROUGHPUT -- EVAL
def tpcc_throughput_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TPCC_THROUGHPUT_DIR)

    for logging_type in LOGGING_TYPES:
        for client_count in CLIENT_COUNTS:

            # RUN EXPERIMENT
            run_experiment(LOGGING,
                           EXPERIMENT_TYPE_THROUGHPUT,
                           logging_type,
                           TPCC_BENCHMARK_TYPE,
                           client_count,
                           DURATION,
                           INVALID_UPDATE_RATIO,
                           DEFAULT_FLUSH_MODE,
                           INVALID_NVM_LATENCY,
                           INVALID_PCOMMIT_LATENCY,
                           DEFAULT_ASYNCHRONOUS_MODE)

            # COLLECT STATS
            collect_stats(TPCC_THROUGHPUT_DIR, TPCC_THROUGHPUT_CSV, TPCC_THROUGHPUT_EXPERIMENT)

# YCSB RECOVERY -- EVAL
def ycsb_recovery_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(YCSB_RECOVERY_DIR)

    client_count = 1
    ycsb_recovery_update_ratio = 1

    for recovery_transaction_count in RECOVERY_DURATIONS:
            for logging_type in LOGGING_TYPES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_RECOVERY,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               client_count,
                               recovery_transaction_count,
                               ycsb_recovery_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE)

                # COLLECT STATS
                collect_stats(YCSB_RECOVERY_DIR, YCSB_RECOVERY_CSV, YCSB_RECOVERY_EXPERIMENT)

# TPCC RECOVERY -- EVAL
def tpcc_recovery_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TPCC_RECOVERY_DIR)

    client_count = 1

    for recovery_transaction_count in RECOVERY_DURATIONS:
            for logging_type in LOGGING_TYPES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_RECOVERY,
                               logging_type,
                               TPCC_BENCHMARK_TYPE,
                               client_count,
                               recovery_transaction_count,
                               INVALID_UPDATE_RATIO,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE)

                # COLLECT STATS
                collect_stats(TPCC_RECOVERY_DIR, TPCC_RECOVERY_CSV, TPCC_RECOVERY_EXPERIMENT)

# YCSB LATENCY -- EVAL
def ycsb_latency_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(YCSB_LATENCY_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for logging_type in LOGGING_TYPES:
            for client_count in CLIENT_COUNTS:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_LATENCY,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               client_count,
                               DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE)

                # COLLECT STATS
                collect_stats(YCSB_LATENCY_DIR, YCSB_LATENCY_CSV, YCSB_LATENCY_EXPERIMENT)

# TPCC LATENCY -- EVAL
def tpcc_latency_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TPCC_LATENCY_DIR)

    for logging_type in LOGGING_TYPES:
        for client_count in CLIENT_COUNTS:

            # RUN EXPERIMENT
            run_experiment(LOGGING,
                           EXPERIMENT_TYPE_LATENCY,
                           logging_type,
                           TPCC_BENCHMARK_TYPE,
                           client_count,
                           DURATION,
                           INVALID_UPDATE_RATIO,
                           DEFAULT_FLUSH_MODE,
                           INVALID_NVM_LATENCY,
                           INVALID_PCOMMIT_LATENCY,
                           DEFAULT_ASYNCHRONOUS_MODE)

            # COLLECT STATS
            collect_stats(TPCC_LATENCY_DIR, TPCC_LATENCY_CSV, TPCC_LATENCY_EXPERIMENT)

# NVM LATENCY -- EVAL
def nvm_latency_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(NVM_LATENCY_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for nvm_logging_type in NVM_LOGGING_TYPES:
            for nvm_latency in NVM_LATENCIES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               nvm_logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               nvm_latency,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE)

                # COLLECT STATS
                collect_stats(NVM_LATENCY_DIR, NVM_LATENCY_CSV, NVM_LATENCY_EXPERIMENT)

# PCOMMIT LATENCY -- EVAL
def pcommit_latency_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(PCOMMIT_LATENCY_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for nvm_logging_type in NVM_LOGGING_TYPES:
            for pcommit_latency in PCOMMIT_LATENCIES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               nvm_logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               pcommit_latency,
                               DEFAULT_ASYNCHRONOUS_MODE)

                # COLLECT STATS
                collect_stats(PCOMMIT_LATENCY_DIR, PCOMMIT_LATENCY_CSV, PCOMMIT_LATENCY_EXPERIMENT)

# FLUSH MODE -- EVAL
def flush_mode_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(FLUSH_MODE_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for nvm_logging_type in NVM_LOGGING_TYPES:
            for flush_mode in FLUSH_MODES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               nvm_logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DURATION,
                               ycsb_update_ratio,
                               flush_mode,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE)

                # COLLECT STATS
                collect_stats(FLUSH_MODE_DIR, FLUSH_MODE_CSV, FLUSH_MODE_EXPERIMENT)


# ASYNCHRONOUS MODE -- EVAL
def asynchronous_mode_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(ASYNCHRONOUS_MODE_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for nvm_logging_type in NVM_LOGGING_TYPES:
            for asynchronous_mode in ASYNCHRONOUS_MODES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               nvm_logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               asynchronous_mode)

                # COLLECT STATS
                collect_stats(ASYNCHRONOUS_MODE_DIR, ASYNCHRONOUS_MODE_CSV, ASYNCHRONOUS_MODE_EXPERIMENT)

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="log",
                                     description='Run Write Behind Logging Experiments',
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50))

    parser.add_argument("-z", "--enable-sdv", help='enable sdv', action='store_true')

    parser.add_argument("-a", "--ycsb_throughput_eval", help='eval ycsb_throughput', action='store_true')
    parser.add_argument("-b", "--tpcc_throughput_eval", help='eval tpcc_throughput', action='store_true')
    parser.add_argument("-c", "--ycsb_recovery_eval", help='eval ycsb_recovery', action='store_true')
    parser.add_argument("-d", "--tpcc_recovery_eval", help='eval tpcc_recovery', action='store_true')
    parser.add_argument("-e", "--ycsb_latency_eval", help='eval ycsb_latency', action='store_true')
    parser.add_argument("-f", "--tpcc_latency_eval", help='eval tpcc_latency', action='store_true')
    parser.add_argument("-g", "--nvm_latency_eval", help='eval nvm_latency', action='store_true')
    parser.add_argument("-i", "--pcommit_latency_eval", help='eval pcommit_latency', action='store_true')
    parser.add_argument("-j", "--flush_mode_eval", help='eval flush_mode', action='store_true')
    parser.add_argument("-k", "--asynchronous_mode_eval", help='eval asynchronous_mode', action='store_true')

    parser.add_argument("-m", "--ycsb_throughput_plot", help='plot ycsb_throughput', action='store_true')
    parser.add_argument("-n", "--tpcc_throughput_plot", help='plot tpcc_throughput', action='store_true')
    parser.add_argument("-o", "--ycsb_recovery_plot", help='plot ycsb_recovery', action='store_true')
    parser.add_argument("-p", "--tpcc_recovery_plot", help='plot tpcc_recovery', action='store_true')
    parser.add_argument("-q", "--ycsb_storage_plot", help='plot ycsb_storage', action='store_true')
    parser.add_argument("-r", "--tpcc_storage_plot", help='plot tpcc_storage', action='store_true')
    parser.add_argument("-s", "--ycsb_latency_plot", help='plot ycsb_latency', action='store_true')
    parser.add_argument("-t", "--tpcc_latency_plot", help='plot tpcc_latency', action='store_true')
    parser.add_argument("-u", "--nvm_latency_plot", help='plot nvm_latency', action='store_true')
    parser.add_argument("-v", "--pcommit_latency_plot", help='plot pcommit_latency', action='store_true')
    parser.add_argument("-w", "--flush_mode_plot", help='plot flush_mode', action='store_true')
    parser.add_argument("-x", "--asynchronous_mode_plot", help='plot asynchronous_mode', action='store_true')

    args = parser.parse_args()

    if args.enable_sdv:
        ENABLE_SDV = os.path.exists(SDV_DIR)
        print("ENABLE_SDV : " + str(ENABLE_SDV))

    ## EVAL

    if args.ycsb_throughput_eval:
        ycsb_throughput_eval()

    if args.tpcc_throughput_eval:
        tpcc_throughput_eval()

    if args.ycsb_recovery_eval:
        ycsb_recovery_eval()

    if args.tpcc_recovery_eval:
        tpcc_recovery_eval()

    if args.ycsb_latency_eval:
        ycsb_latency_eval()

    if args.tpcc_latency_eval:
        tpcc_latency_eval()

    if args.nvm_latency_eval:
        nvm_latency_eval()

    if args.pcommit_latency_eval:
        pcommit_latency_eval()

    if args.flush_mode_eval:
        flush_mode_eval()

    if args.asynchronous_mode_eval:
        asynchronous_mode_eval()

    ## PLOT

    if args.ycsb_throughput_plot:
        ycsb_throughput_plot()

    if args.tpcc_throughput_plot:
        tpcc_throughput_plot()

    if args.ycsb_recovery_plot:
        ycsb_recovery_plot()

    if args.tpcc_recovery_plot:
        tpcc_recovery_plot()

    if args.ycsb_storage_plot:
        ycsb_storage_plot()

    if args.tpcc_storage_plot:
        tpcc_storage_plot()

    if args.ycsb_latency_plot:
        ycsb_latency_plot()

    if args.tpcc_latency_plot:
        tpcc_latency_plot()

    if args.nvm_latency_plot:
        nvm_latency_plot()

    if args.pcommit_latency_plot:
        pcommit_latency_plot()

    if args.flush_mode_plot:
        flush_mode_plot()

    if args.asynchronous_mode_plot:
        asynchronous_mode_plot()

    #create_legend_logging_types()
    #create_legend_storage()
