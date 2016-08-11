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
COLOR_MAP = ('#5D8AC6', '#9FD35D', '#C02500', "#003557", "#538B00", "#920000")
#COLOR_MAP = ('#9EC9E9', '#80CA86', '#F58A87', "#5DA5DA", "#66A26B", "#F15854")


OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = ([ "////", "o", "\\\\", "////", "o", "\\\\", "//////", "." , "\\\\\\"])

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = COLOR_MAP
OPT_LINE_WIDTH = 6.0
OPT_MARKER_SIZE = 10.0
DATA_LABELS = []

OPT_STACK_COLORS = ('#BBBB88', '#EEDD99', '#EE8899', '#EEC290')
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

NUMACTL = "numactl"
NUMACTL_FLAGS = "--membind=2"

PERF_LOCAL = "/usr/bin/perf"
PERF = "/usr/lib/linux-tools/3.11.0-12-generic/perf"

NVM_LATENCIES = ("160", "320", "640")
NVM_LATENCIES_LABELS = ("1X", "2X", "4X")
DEFAULT_NVM_LATENCY = NVM_LATENCIES[0]
INVALID_NVM_LATENCY = 0
INVALID_TRANSACTION_COUNT = 0
INVALID_DURATION = 0

PCOMMIT_LATENCIES = ("0", "10", "100", "1000", "10000", "100000")
PCOMMIT_LABELS = ("Current", "10", "100", "1000", "10000", "100000")
INVALID_PCOMMIT_LATENCY = 0

OUTPUT_FILE = "outputfile.summary"

# Refer LoggingType in common/types.h
LOGGING_TYPES = (1, 2, 3, 4, 5, 6)
LOGGING_NAMES = ("nvm-wbl", "ssd-wbl", "hdd-wbl", "nvm-wal", "ssd-wal", "hdd-wal")

NVM_LOGGING_TYPES = (1, 4)
NVM_LOGGING_NAMES = ("nvm-wbl", "nvm-wal")

WAL_LOGGING_TYPES = (4, 5, 6)
WAL_LOGGING_NAMES = ("nvm-wal", "ssd-wal", "hdd-wal")

SCALE_FACTOR = 1
DATABASE_FILE_SIZE = 4096  # DATABASE FILE SIZE (MB)

DEFAULT_DURATION = 1000

CLIENT_COUNTS = (1, 2, 4, 8)
DEFAULT_CLIENT_COUNT = 8

YCSB_BENCHMARK_TYPE = 1
TPCC_BENCHMARK_TYPE = 2

YCSB_RECOVERY_COUNTS = (10000, 100000)
TPCC_RECOVERY_COUNTS = (1000, 10000)
YCSB_RECOVERY_COUNTS_NAMES = ("10000", "100000")
TPCC_RECOVERY_COUNTS_NAMES = ("1000", "10000")

YCSB_UPDATE_RATIOS = (0.1, 0.5, 0.9)
YCSB_UPDATE_NAMES = ("read-heavy", "balanced", "write-heavy")
INVALID_UPDATE_RATIO = 0

FLUSH_MODES = ("1", "2")
DEFAULT_FLUSH_MODE = 2
FLUSH_MODES_NAMES = ("CLFLUSH", "CLWB")

ASYNCHRONOUS_MODES = ("1", "4", "3")
DEFAULT_ASYNCHRONOUS_MODE = 1
ASYNCHRONOUS_MODES_NAMES = ("Enabled", "No Writes", "Disabled")

REPLICATION_MODES = ("1", "2", "3", "4")
REPLICATION_MODES_NAMES = ("Disabled", "Async", "Semi-Sync", "Sync")

GROUP_COMMIT_INTERVALS = ("10", "100", "1000", "10000", "100000")
DEFAULT_GROUP_COMMIT_INTERVAL = ("200")

OPS_COUNT = ("1", "10", "100", "1000")
DEFAULT_OPS_COUNT = 1

LONG_RUNNING_TXN_COUNTS = ("100", "1000", "10000", "100000", "1000000")
DEFAULT_LONG_RUNNING_TXN_COUNT = ("0")

ABORT_MODES = ("0", "1")
ABORT_MODE_NAMES = ("commit", "abort")
DEFAULT_ABORT_MODE = "0"

DEFAULT_GOETZ_MODE = "0"
REDO_LENGTHS = ("10", "100")
DEFAULT_REDO_LENGTH = ("0")
REDO_FRACTIONS = ("0.001", "0.01", "0.1", "1")
DEFAULT_REDO_FRACTION = ("0")

HYBRID_STORAGE_RATIOS = ("0", "0.1", "0.5", "1")
DEFAULT_STORAGE_RATIO = "0"

EXPERIMENT_TYPE_THROUGHPUT = 1
EXPERIMENT_TYPE_RECOVERY = 2
EXPERIMENT_TYPE_STORAGE = 3
EXPERIMENT_TYPE_LATENCY = 4

STORAGE_LOGGING_TYPES = ("DRAM", "NVM", "DRAM", "NVM")
STORAGE_LABELS = ("Table Heap", "Index", "Log", "Checkpoint")

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

MOTIVATION_DIR = BASE_DIR + "/results/motivation/"
MOTIVATION_EXPERIMENT = 11
MOTIVATION_THROUGHPUT_CSV = "motivation_throughput.csv"
MOTIVATION_RECOVERY_CSV = "motivation_recovery.csv"
MOTIVATION_STORAGE_CSV = "motivation_storage.csv"

REPLICATION_THROUGHPUT_DIR = BASE_DIR + "/results/replication/throughput/"
REPLICATION_THROUGHPUT_EXPERIMENT = 12
REPLICATION_THROUGHPUT_CSV = "replication_throughput.csv"

REPLICATION_LATENCY_DIR = BASE_DIR + "/results/replication/latency/"
REPLICATION_LATENCY_EXPERIMENT = 13
REPLICATION_LATENCY_CSV = "replication_latency.csv"

GROUP_COMMIT_DIR = BASE_DIR + "/results/group_commit/"
GROUP_COMMIT_EXPERIMENT = 14
GROUP_COMMIT_CSV = "group_commit.csv"

TIME_TO_COMMIT_DIR = BASE_DIR + "/results/time_to_commit/"
TIME_TO_COMMIT_EXPERIMENT = 15
TIME_TO_COMMIT_CSV = "time_to_commit.csv"

LONG_RUNNING_TXN_DIR = BASE_DIR + "/results/long_running_txn/"
LONG_RUNNING_TXN_EXPERIMENT = 16
LONG_RUNNING_TXN_CSV = "long_running_txn.csv"

GOETZ_DIR = BASE_DIR + "/results/goetz/"
GOETZ_EXPERIMENT = 17
GOETZ_CSV = "goetz.csv"

HYBRID_DIR = BASE_DIR + "/results/hybrid/"
HYBRID_EXPERIMENT = 18
HYBRID_CSV = "hybrid.csv"

###################################################################################
# UTILS
###################################################################################

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def loadDataFile(n_rows, n_cols, path):
    data_file = open(path, "r")
    reader = csv.reader(data_file)

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

def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups

def add_line(ax, xpos, ypos):
    line = plot.Line2D([xpos, xpos], [ypos + .1, ypos - 0.1],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

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

# Figure out abort mode name
def getAbortModeName(abort_mode):

    abort_mode_offset = ABORT_MODES.index(str(abort_mode))
    abort_name = ABORT_MODE_NAMES[abort_mode_offset]

    return abort_name

###################################################################################
# PLOT
###################################################################################

def create_legend_logging_types(line_chart, nvm_only):
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    if nvm_only == True:
        figlegend = pylab.figure(figsize=(5.25, 0.5))
    else:
        figlegend = pylab.figure(figsize=(15, 0.5))

    if nvm_only == False:
        ARRAY = LOGGING_NAMES
    else:
        ARRAY = NVM_LOGGING_NAMES

    N = len(ARRAY);
    ind = np.arange(1)
    margin = 0.10
    width = ((1.0 - 2 * margin) / N) * 2
    data = [1]
    x_values = [1]

    bars = [None] * (len(ARRAY) + 1) * 2
    lines = [None] * (len(ARRAY) + 1) * 2

    idx = 0
    for group in xrange(len(ARRAY)):
        color_idx = idx
        if nvm_only == True:
            if idx == 1:
                color_idx = 3

        bars[idx] = ax1.bar(ind + margin + ((idx + 1) * width), data, width,
                            color=OPT_COLORS[color_idx],
                            linewidth=BAR_LINEWIDTH,
                            hatch=OPT_PATTERNS[group])

        idx = idx + 1

    idx = 0
    for group in xrange(len(ARRAY)):
        color_idx = idx
        if nvm_only == True:
            if idx == 1:
                color_idx = 3

        lines[idx], = ax1.plot(x_values, data, color=OPT_LINE_COLORS[color_idx],
                               linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[color_idx],
                               markersize=OPT_MARKER_SIZE,
                               label=str(group))

        idx = idx + 1

    ARRAY_UPPER_CASE = [x.upper() for x in ARRAY]

    # LEGEND
    if line_chart == False:
        figlegend.legend(bars, ARRAY_UPPER_CASE, prop=LEGEND_FP,
                         loc=1, ncol=len(ARRAY),
                         mode="expand", shadow=OPT_LEGEND_SHADOW,
                         frameon=False, borderaxespad=0.0,
                         handleheight=1, handlelength=4)
    else:
        figlegend.legend(lines,  ARRAY_UPPER_CASE, prop=LEGEND_FP,
                         loc=1, ncol=len(ARRAY),
                         mode="expand", shadow=OPT_LEGEND_SHADOW,
                         frameon=False, borderaxespad=0.0,
                         handleheight=1, handlelength=4)

    filename = 'legend_logging_types'
    if line_chart == True:
        filename = filename + "_line"
    if nvm_only == True:
        filename = filename + "_nvm"
    filename = filename + ".pdf"

    figlegend.savefig(filename)

def create_legend_storage():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(9, 0.5))

    num_items = 5;
    ind = np.arange(1)
    margin = 0.10
    width = (1.0 - 2 * margin) / num_items

    bars = [None] * len(STORAGE_LABELS) * 2

    for group in xrange(len(STORAGE_LABELS)):
        data = [1]
        bars[group] = ax1.bar(ind + margin + (group * width), data, width,
                              color=OPT_STACK_COLORS[group],
                              linewidth=BAR_LINEWIDTH)

    # LEGEND
    figlegend.legend(bars, STORAGE_LABELS, prop=LABEL_FP,
                     loc=1, ncol=len(STORAGE_LABELS),
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_storage.pdf')

def create_legend_update_ratio():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(9, 0.5))
    idx = 0
    lines = [None] * len(YCSB_UPDATE_NAMES)

    workload_mix = ("Read-Heavy", "Balanced", "Write-Heavy")

    for group in xrange(len(YCSB_UPDATE_NAMES)):
        data = [1]
        x_values = [1]

        lines[idx], = ax1.plot(x_values, data, color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # LEGEND
    figlegend.legend(lines,  workload_mix, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_update_ratio.pdf')

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
                 color=OPT_COLORS[idx],
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
    ax1.set_xlabel("Number of Worker Threads", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_ycsb_recovery_bar_chart(datasets, ycsb):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in YCSB_RECOVERY_COUNTS]
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
                                      linewidth=BAR_LINEWIDTH,
                                      hatch=OPT_PATTERNS[group])

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Recovery Time (s)", fontproperties=LABEL_FP)
    ax1.set_yscale('log', nonposy='clip')
    ax1.tick_params(axis='y', which='minor', left='off', right='off')
    YLIMIT_MIN = math.pow(10, -2)
    YLIMIT_MAX = math.pow(10, +2)
    ax1.set_ylim(YLIMIT_MIN, YLIMIT_MAX)
    ax1.set_yticklabels(["", "0.1", "1", "10", "100", "1000"])

    # X-AXIS
    ax1.set_xlabel("Number of Transactions (K)", fontproperties=LABEL_FP)
    ax1.set_xticks(ind + margin + (group * width)/2.0 )

    if ycsb == True:
        ax1.set_xticklabels(YCSB_RECOVERY_COUNTS_NAMES)
    else:
        ax1.set_xticklabels(TPCC_RECOVERY_COUNTS_NAMES)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_ycsb_storage_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    width = 1.0

    # GROUP LABELS
    group_labels = {
            'NVM-WBL':
                {'Milk': 10,
                'Water': 20},
            'NVM-WAL':
                {'Sugar': 5,
                'Honey': 6}
           }

    groups = mk_groups(group_labels)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    scale = 1. / ly
    for pos in xrange(ly + 1):
        add_line(ax1, pos * scale, -.1)
    ypos = -.2

    NVM_LOGGING_NAMES_UPPER_CASE = [x.upper() for x in NVM_LOGGING_NAMES]
    while groups:
        group = groups.pop()
        pos = 0
        idx = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            ax1.text(lxpos, ypos - 0.1, NVM_LOGGING_NAMES_UPPER_CASE[idx],
                     ha='center', transform=ax1.transAxes,
                     fontproperties = LABEL_FP)
            add_line(ax1, pos * scale, ypos)
            pos += rpos
            idx = idx + 1

        add_line(ax1, pos * scale, ypos)
        ypos -= .1

    col_offset = 0.2
    col_width = width - col_offset
    xbarticks = [x + col_offset/2.0 for x in xticks]
    xlabelticks = [x + 0.5 for x in xticks]

    bars = [None] * len(STORAGE_LABELS) * 2

    datasets = map(list, map(None,*datasets))

    # TYPE
    bottom_list = [0] * len(datasets[0])
    for group_type in  xrange(1, len(datasets)):
        LOG.info("TYPE :: %d %s", group_type, datasets[group_type])

        bars[group_type] = ax1.bar(xbarticks, datasets[group_type], col_width,
                             color=OPT_STACK_COLORS[group_type - 1], linewidth=BAR_LINEWIDTH,
                             bottom = bottom_list)
        bottom_list = map(add, bottom_list, datasets[group_type])

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.set_ylabel("Storage (GB)", fontproperties=LABEL_FP)
    ax1.yaxis.set_major_locator(MaxNLocator(4))

    # X-AXIS
    ax1.tick_params(axis='x', which='both', top='off', bottom='off')
    ax1.set_xticks(xlabelticks)
    ax1.set_xticklabels(STORAGE_LOGGING_TYPES)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(LABEL_FP)

    return (fig)

def create_ycsb_latency_bar_chart(datasets, experiment_type):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in CLIENT_COUNTS]
    N = len(x_labels)
    M = len(LOGGING_NAMES)
    ind = np.arange(N)
    margin = 0.1
    width = (1.0 - 2 * margin) / M
    bars = [None] * M * N

    for group in xrange(len(datasets)):
        # GROUP
        group_data = []

        for line in  xrange(len(datasets[group])):
            for col in xrange(len(datasets[group][line])):
                if col == 1:
                    group_data.append(datasets[group][line][col] * 1000) # ms

        LOG.info("group_data = %s", str(group_data))

        bars[group] = ax1.bar(ind + margin + (group * width), group_data, width,
                              color=OPT_COLORS[group],
                              linewidth=BAR_LINEWIDTH,
                              hatch=OPT_PATTERNS[group])

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)
    ax1.set_yscale('log', nonposy='clip')
    ax1.tick_params(axis='y', which='minor', left='off', right='off')

    if experiment_type == "ycsb":
        YLIMIT_MIN = math.pow(10, -2)
        YLIMIT_MAX = math.pow(10, +2)
        ax1.set_ylim(YLIMIT_MIN, YLIMIT_MAX)
        ax1.set_yticklabels(["", "0.01", "0.1", "1", "10", "100"])
    elif experiment_type == "tpcc":
        YLIMIT_MIN = math.pow(10, -1)
        YLIMIT_MAX = math.pow(10, +3)
        ax1.set_ylim(YLIMIT_MIN, YLIMIT_MAX)
        ax1.set_yticklabels(["", "0.1", "1", "10", "100", "1000"])

    # X-AXIS
    ax1.set_xticks(ind + margin + 0.5)
    ax1.set_xlabel("Number of Worker Threads", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_nvm_latency_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in NVM_LATENCIES]
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

        color_idx = 0
        if idx == 1:
            color_idx = 3

        ax1.plot(ind + 0.5, group_data,
                 color=OPT_COLORS[color_idx],
                 linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[color_idx], markersize=OPT_MARKER_SIZE,
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
    ax1.set_xlabel("NVM Latency", fontproperties=LABEL_FP)
    ax1.set_xticklabels(NVM_LATENCIES_LABELS)
    ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_pcommit_latency_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in PCOMMIT_LATENCIES]
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
    ax1.set_xlabel("PCOMMIT Latency (ns)", fontproperties=LABEL_FP)
    ax1.set_xticklabels(PCOMMIT_LABELS)

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

        color_group = 0
        if group == 1:
            color_group = 3

        bars[group] = ax1.bar(ind + margin + (group * width), group_data, width,
                                      color=OPT_COLORS[color_group],
                                      linewidth=BAR_LINEWIDTH,
                                      hatch=OPT_PATTERNS[group])


    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Throughput", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("NVM Flush Instruction", fontproperties=LABEL_FP)
    ax1.set_xticklabels(FLUSH_MODES_NAMES)

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

        color_group = group

        bars[group] = ax1.bar(ind + margin + (group * width), group_data, width,
                                      color=OPT_COLORS[color_group],
                                      linewidth=BAR_LINEWIDTH,
                                      hatch=OPT_PATTERNS[group])


    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS + 2))
    ax1.minorticks_off()
    ax1.set_ylabel("Throughput", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Logging Status", fontproperties=LABEL_FP)
    ax1.set_xticklabels(ASYNCHRONOUS_MODES_NAMES)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_motivation_bar_chart(datasets, bar_type):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    N = 1
    M = len(NVM_LOGGING_NAMES)
    ind = np.arange(N)
    margin = 0.15
    width = ((1.0 - 3 * margin) / M)
    bars = [None] * M * N
    label_locations = [0.30, 0.71]

    NVM_LOGGING_NAMES_UPPER_CASE = [x.upper() for x in NVM_LOGGING_NAMES]

    for group in xrange(len(datasets)):
        # GROUP
        group_data = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    group_data.append(datasets[group][line][col])

        LOG.info("group_data = %s", str(group_data))

        color_group = 0
        offset_group = 1
        if group == 1:
            color_group = 3
            offset_group = 2

        bars[group] = ax1.bar(ind + (offset_group * margin) + (group * width), group_data, width,
                                      color=OPT_COLORS[color_group],
                                      linewidth=BAR_LINEWIDTH,
                                      hatch=OPT_PATTERNS[color_group])

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()

    if bar_type == "Throughput":
        ax1.set_ylabel("Throughput", fontproperties=LABEL_FP)
    elif bar_type == "Recovery":
        ax1.set_ylabel("Recovery Latency (s)", fontproperties=LABEL_FP)
        ax1.set_yscale('log', nonposy='clip')
        ax1.tick_params(axis='y', which='minor', left='off', right='off')
        ax1.set_yticklabels(["", "0.1", "1", "10", "100", "1000"])
    elif bar_type == "Storage":
        ax1.set_ylabel("Storage (GB)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xlabel("Logging Protocol", fontproperties=LABEL_FP)
    ax1.set_xticks(label_locations)
    ax1.set_xticklabels(NVM_LOGGING_NAMES_UPPER_CASE)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_replication_chart(datasets, experiment_type):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in REPLICATION_MODES]
    N = len(x_labels)
    M = len(NVM_LOGGING_NAMES)
    ind = np.arange(N)
    margin = 0.15
    width = ((1.0 - 2 * margin) / M)
    bars = [None] * M * N

    REPLICATION_MODES_NAMES_UPPER_CASE = [x.upper() for x in REPLICATION_MODES_NAMES]

    for group in xrange(len(datasets)):
        # GROUP
        group_data = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    if experiment_type == "Throughput":
                        group_data.append(datasets[group][line][col])
                    elif experiment_type == "Latency":
                        group_data.append(datasets[group][line][col] * 1000) # ms

        LOG.info("group_data = %s", str(group_data))

        color_group = 0
        if group == 1:
            color_group = 3

        bars[group] = ax1.bar(ind + margin + (group * width), group_data, width,
                                      color=OPT_COLORS[color_group],
                                      linewidth=BAR_LINEWIDTH,
                                      hatch=OPT_PATTERNS[group])


    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()

    if experiment_type == "Throughput":
        ax1.set_ylabel("Throughput", fontproperties=LABEL_FP)
    elif experiment_type == "Latency":
        ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Replication Mode", fontproperties=LABEL_FP)
    ax1.set_xticklabels(REPLICATION_MODES_NAMES_UPPER_CASE)

    if experiment_type == "Throughput":
        ax1.set_xlim([0.01, N - 0.01])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_group_commit_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in GROUP_COMMIT_INTERVALS]
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
                 color=OPT_COLORS[idx],
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
    ax1.set_xlabel("Group Commit Interval", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    #ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_time_to_commit_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in OPS_COUNT]
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
                 color=OPT_COLORS[idx],
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
    ax1.set_xlabel("Op Count", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    #ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_long_running_txn_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in LONG_RUNNING_TXN_COUNTS]
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
                 color=OPT_COLORS[idx],
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
    ax1.set_xlabel("Long Running Transaction Count", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    #ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_goetz_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_labels = [str(i) for i in REDO_FRACTIONS]
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
                 color=OPT_COLORS[idx],
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
    ax1.set_xlabel("Redo Fraction", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    #ax1.set_xlim([0.25, N - 0.25])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_hybrid_line_chart(datasets):
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
                 color=OPT_COLORS[idx],
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
    ax1.set_xlabel("Number of Client Threads", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    #ax1.set_xlim([0.25, N - 0.25])

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

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

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

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

# YCSB RECOVERY -- PLOT
def ycsb_recovery_plot():

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = YCSB_RECOVERY_DIR + "/" + logging_name + "/" + YCSB_RECOVERY_CSV

        dataset = loadDataFile(len(YCSB_RECOVERY_COUNTS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_recovery_bar_chart(datasets, True)

    fileName = "ycsb-" + "recovery" + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# TPCC RECOVERY -- PLOT
def tpcc_recovery_plot():

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = TPCC_RECOVERY_DIR + "/" + logging_name + "/" + TPCC_RECOVERY_CSV

        dataset = loadDataFile(len(TPCC_RECOVERY_COUNTS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_recovery_bar_chart(datasets, False)

    fileName = "tpcc-" + "recovery" + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# YCSB STORAGE -- PLOT
def ycsb_storage_plot():

    data_file =  os.path.realpath(os.path.join(YCSB_STORAGE_DIR, YCSB_STORAGE_CSV))

    dataset = loadDataFile(len(STORAGE_LOGGING_TYPES), len(STORAGE_LABELS) + 1, data_file)

    fig = create_ycsb_storage_bar_chart(dataset)

    fileName = "ycsb-storage.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# TPCC STORAGE -- PLOT
def tpcc_storage_plot():

    data_file =  os.path.realpath(os.path.join(TPCC_STORAGE_DIR, TPCC_STORAGE_CSV))

    dataset = loadDataFile(len(STORAGE_LOGGING_TYPES), len(STORAGE_LABELS) + 1, data_file)

    fig = create_ycsb_storage_bar_chart(dataset)

    fileName = "tpcc-storage.pdf"

    saveGraph(fig, fileName, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

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

        fig = create_ycsb_latency_bar_chart(datasets , "ycsb")

        fileName = "ycsb-" + "latency-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

# TPCC LATENCY -- PLOT
def tpcc_latency_plot():

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = TPCC_LATENCY_DIR + "/" + logging_name + "/" + TPCC_LATENCY_CSV

        dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_latency_bar_chart(datasets, "tpcc")

    fileName = "tpcc-" + "latency" + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

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

        fig = create_nvm_latency_line_chart(datasets)

        fileName = "nvm-latency-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH/1.5, height=OPT_GRAPH_HEIGHT/1.5)

# PCOMMIT LATENCY -- PLOT
def pcommit_latency_plot():

    datasets = []
    nvm_wbl_logging_type = 1

    # figure out logging name and ycsb update name
    nvm_logging_name = getLoggingName(nvm_wbl_logging_type)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        data_file = PCOMMIT_LATENCY_DIR + "/" + ycsb_update_name + "/" + nvm_logging_name + "/" + PCOMMIT_LATENCY_CSV

        dataset = loadDataFile(len(PCOMMIT_LATENCIES), 2, data_file)
        datasets.append(dataset)

    fig = create_pcommit_latency_line_chart(datasets)

    fileName = "pcommit-latency.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH*1.5, height=OPT_GRAPH_HEIGHT/3.0)

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

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH/1.5, height=OPT_GRAPH_HEIGHT/1.5)

# ASYNCHRONOUS MODE -- PLOT
def asynchronous_mode_plot():

    async_update_ratio = YCSB_UPDATE_RATIOS[2]

    ycsb_update_name = getYCSBUpdateName(async_update_ratio)

    datasets = []
    for logging_type in LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = ASYNCHRONOUS_MODE_DIR + "/" + ycsb_update_name + "/" + logging_name + "/" + ASYNCHRONOUS_MODE_CSV

        dataset = loadDataFile(len(ASYNCHRONOUS_MODES), 2, data_file)
        datasets.append(dataset)

    fig = create_asynchronous_mode_bar_chart(datasets)

    fileName = "asynchronous-mode-" + ycsb_update_name + ".pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH*1.5, height=OPT_GRAPH_HEIGHT/2.0)

# MOTIVATION -- PLOT
def motivation_plot():

    datasets = []
    for logging_type in NVM_LOGGING_TYPES:

        # figure out logging name and ycsb update name
        nvm_logging_name = getLoggingName(logging_type)

        data_file = MOTIVATION_DIR + "/" + nvm_logging_name + "/" + MOTIVATION_THROUGHPUT_CSV

        dataset = loadDataFile(1, 2, data_file)
        datasets.append(dataset)

    fig = create_motivation_bar_chart(datasets, "Throughput")

    fileName = "motivation-" + "throughput.pdf"
    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH/1.5, height=OPT_GRAPH_HEIGHT/1.5)

    datasets = []
    for logging_type in NVM_LOGGING_TYPES:

        # figure out logging name and ycsb update name
        nvm_logging_name = getLoggingName(logging_type)

        data_file = MOTIVATION_DIR + "/" + nvm_logging_name + "/" + MOTIVATION_RECOVERY_CSV

        dataset = loadDataFile(1, 2, data_file)
        datasets.append(dataset)

    fig = create_motivation_bar_chart(datasets, "Recovery")

    fileName = "motivation-" + "recovery.pdf"
    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH/1.5, height=OPT_GRAPH_HEIGHT/1.5)

    datasets = []
    for logging_type in NVM_LOGGING_TYPES:

        # figure out logging name and ycsb update name
        nvm_logging_name = getLoggingName(logging_type)

        data_file = MOTIVATION_DIR + "/" + nvm_logging_name + "/" + MOTIVATION_STORAGE_CSV

        dataset = loadDataFile(1, 2, data_file)
        datasets.append(dataset)

    fig = create_motivation_bar_chart(datasets, "Storage")

    fileName = "motivation-" + "storage.pdf"
    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH/1.5, height=OPT_GRAPH_HEIGHT/1.5)

# REPLICATION -- PLOT
def replication_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for logging_type in NVM_LOGGING_TYPES:

            # figure out logging name and ycsb update name
            nvm_logging_name = getLoggingName(logging_type)

            data_file = REPLICATION_THROUGHPUT_DIR + "/" + ycsb_update_name + "/" + nvm_logging_name + "/" + REPLICATION_THROUGHPUT_CSV

            dataset = loadDataFile(len(REPLICATION_MODES), 2, data_file)
            datasets.append(dataset)

        fig = create_replication_chart(datasets, "Throughput")

        fileName = "replication-" + "throughput-" + ycsb_update_name + ".pdf"
        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for logging_type in NVM_LOGGING_TYPES:

            # figure out logging name and ycsb update name
            nvm_logging_name = getLoggingName(logging_type)

            data_file = REPLICATION_LATENCY_DIR + "/" + ycsb_update_name + "/" + nvm_logging_name + "/" + REPLICATION_LATENCY_CSV

            dataset = loadDataFile(len(REPLICATION_MODES), 2, data_file)
            datasets.append(dataset)

        fig = create_replication_chart(datasets, "Latency")

        fileName = "replication-" + "latency-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

# GROUP COMMIT -- PLOT
def group_commit_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for logging_type in LOGGING_TYPES:

            # figure out logging name and ycsb update name
            logging_name = getLoggingName(logging_type)

            data_file = GROUP_COMMIT_DIR + "/" + ycsb_update_name + "/" + logging_name + "/" + GROUP_COMMIT_CSV

            dataset = loadDataFile(len(GROUP_COMMIT_INTERVALS), 2, data_file)
            datasets.append(dataset)

        fig = create_group_commit_line_chart(datasets)

        fileName = "group-commit-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

# TIME TO COMMIT -- PLOT
def time_to_commit_plot():


    for abort_mode in ABORT_MODES:

        datasets = []
        for logging_type in LOGGING_TYPES:

            # figure out logging name and ycsb update name
            logging_name = getLoggingName(logging_type)

            data_file = TIME_TO_COMMIT_DIR + "/" + getAbortModeName(abort_mode) + "/" + logging_name + "/" + TIME_TO_COMMIT_CSV

            dataset = loadDataFile(len(OPS_COUNT), 2, data_file)
            datasets.append(dataset)

        fig = create_time_to_commit_line_chart(datasets)

        fileName = "time-to-" + getAbortModeName(abort_mode) + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

# LONG RUNNING TXN -- PLOT
def long_running_txn_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for logging_type in LOGGING_TYPES:

            # figure out logging name and ycsb update name
            logging_name = getLoggingName(logging_type)

            data_file = LONG_RUNNING_TXN_DIR + "/" + ycsb_update_name + "/" + logging_name + "/" + LONG_RUNNING_TXN_CSV

            dataset = loadDataFile(len(LONG_RUNNING_TXN_COUNTS), 2, data_file)
            datasets.append(dataset)

        fig = create_long_running_txn_line_chart(datasets)

        fileName = "long-running-txn-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

# GOETZ -- PLOT
def goetz_plot():

    for redo_length in REDO_LENGTHS:

        for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
    
            ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)
    
            datasets = []
            for logging_type in WAL_LOGGING_TYPES:
    
                # figure out logging name and ycsb update name
                logging_name = getLoggingName(logging_type)
    
                data_file = GOETZ_DIR + "/" + str(redo_length) + "/" + ycsb_update_name + "/" + logging_name + "/" + GOETZ_CSV
    
                dataset = loadDataFile(len(REDO_FRACTIONS), 2, data_file)
                datasets.append(dataset)
    
            fig = create_goetz_line_chart(datasets)
    
            fileName = "goetz-" + ycsb_update_name + "-" + str(redo_length) + ".pdf"
    
            saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)

# HYBRID -- PLOT
def hybrid_plot():

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:

        ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)

        datasets = []
        for hybrid_storage_ratio in HYBRID_STORAGE_RATIOS:

            data_file = HYBRID_DIR + "/" + str(hybrid_storage_ratio) + "/" + ycsb_update_name + "/" + HYBRID_CSV

            dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
            datasets.append(dataset)

        fig = create_hybrid_line_chart(datasets)

        fileName = "hybrid-" + ycsb_update_name + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/3.0)            

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
                   asynchronous_mode,
                   transaction_count,
                   group_commit_interval,
                   op_count,
                   abort_mode,
                   long_running_txn_count,
                   goetz_mode,
                   redo_length,
                   redo_fraction,
                   hybrid_storage_ratio):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    # With numactl or not
    if ENABLE_SDV:
        subprocess.call([NUMACTL,
                         NUMACTL_FLAGS,
                         program,
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
                         "-a", str(asynchronous_mode),
                         "-t", str(transaction_count),
                         "-w", str(group_commit_interval),
                         "-o", str(op_count),
                         "-r", str(abort_mode),
                         "-q", str(long_running_txn_count),
                         "-s", str(goetz_mode),
                         "-m", str(redo_length),
                         "-j", str(redo_fraction),
                         "-x", str(hybrid_storage_ratio)])
    else:
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
                         "-a", str(asynchronous_mode),
                         "-t", str(transaction_count),
                         "-w", str(group_commit_interval),
                         "-o", str(op_count),
                         "-r", str(abort_mode),
                         "-q", str(long_running_txn_count),
                         "-s", str(goetz_mode),
                         "-m", str(redo_length),
                         "-j", str(redo_fraction),
                         "-x", str(hybrid_storage_ratio)])


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
        wait_timeout = data[10]
        op_count = data[11]
        abort_mode = data[12]
        long_running_txn_count = data[13]
        goetz_mode = data[14]
        redo_fraction = data[15]
        redo_length = data[16]
        hybrid_storage_ratio = data[17]

        stat = data[18]

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        # MAKE RESULTS FILE DIR
        if category == YCSB_THROUGHPUT_EXPERIMENT or category == YCSB_LATENCY_EXPERIMENT \
         or category == GROUP_COMMIT_EXPERIMENT or category == LONG_RUNNING_TXN_EXPERIMENT:
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
        elif category == TIME_TO_COMMIT_EXPERIMENT:
            result_directory = result_dir + "/" + getAbortModeName(int(abort_mode)) + "/" + logging_name
        elif category == GOETZ_EXPERIMENT:
            result_directory = result_dir + "/" + str(redo_length) + "/" + ycsb_update_name + "/" + logging_name
        elif category == HYBRID_EXPERIMENT:
            ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)
            result_directory = result_dir + "/" + str(hybrid_storage_ratio) + "/" + ycsb_update_name

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
        elif category == GROUP_COMMIT_EXPERIMENT:
            result_file.write(str(wait_timeout) + " , " + str(stat) + "\n")
        elif category == TIME_TO_COMMIT_EXPERIMENT:
            result_file.write(str(op_count) + " , " + str(stat) + "\n")
        elif category == LONG_RUNNING_TXN_EXPERIMENT:
            result_file.write(str(long_running_txn_count) + " , " + str(stat) + "\n")
        elif category == GOETZ_EXPERIMENT:
            result_file.write(str(redo_fraction) + " , " + str(stat) + "\n")
        elif category == HYBRID_EXPERIMENT:
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
        LOG.info("Setting nvm latency to : " + str(nvm_latency));
        subprocess.call(['sudo', SDV_SCRIPT, '--enable', '--pm-latency', str(nvm_latency)], stdout=FNULL)
        os.chdir(cwd)
        FNULL.close()

def reset_nvm_latency():
    if ENABLE_SDV :
        FNULL = open(os.devnull, 'w')
        cwd = os.getcwd()
        os.chdir(SDV_DIR)
        LOG.info("Resetting nvm latency to : " + str(DEFAULT_NVM_LATENCY));
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
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

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
                           DEFAULT_DURATION,
                           INVALID_UPDATE_RATIO,
                           DEFAULT_FLUSH_MODE,
                           INVALID_NVM_LATENCY,
                           INVALID_PCOMMIT_LATENCY,
                           DEFAULT_ASYNCHRONOUS_MODE,
                           INVALID_TRANSACTION_COUNT,
                           DEFAULT_GROUP_COMMIT_INTERVAL,
                           DEFAULT_OPS_COUNT,
                           DEFAULT_ABORT_MODE,
                           DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                           DEFAULT_GOETZ_MODE,
                           DEFAULT_REDO_LENGTH,
                           DEFAULT_REDO_FRACTION,
                           DEFAULT_STORAGE_RATIO)

            # COLLECT STATS
            collect_stats(TPCC_THROUGHPUT_DIR, TPCC_THROUGHPUT_CSV, TPCC_THROUGHPUT_EXPERIMENT)

# YCSB RECOVERY -- EVAL
def ycsb_recovery_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(YCSB_RECOVERY_DIR)

    client_count = 8
    ycsb_recovery_update_ratio = 1

    for recovery_transaction_count in YCSB_RECOVERY_COUNTS:
            for logging_type in LOGGING_TYPES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_RECOVERY,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               client_count,
                               INVALID_DURATION,
                               ycsb_recovery_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               recovery_transaction_count,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

                # COLLECT STATS
                collect_stats(YCSB_RECOVERY_DIR, YCSB_RECOVERY_CSV, YCSB_RECOVERY_EXPERIMENT)

# TPCC RECOVERY -- EVAL
def tpcc_recovery_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TPCC_RECOVERY_DIR)

    client_count = 8

    for recovery_transaction_count in TPCC_RECOVERY_COUNTS:
            for logging_type in LOGGING_TYPES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_RECOVERY,
                               logging_type,
                               TPCC_BENCHMARK_TYPE,
                               client_count,
                               INVALID_DURATION,
                               INVALID_UPDATE_RATIO,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               recovery_transaction_count,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

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
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

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
                           DEFAULT_DURATION,
                           INVALID_UPDATE_RATIO,
                           DEFAULT_FLUSH_MODE,
                           INVALID_NVM_LATENCY,
                           INVALID_PCOMMIT_LATENCY,
                           DEFAULT_ASYNCHRONOUS_MODE,
                           INVALID_TRANSACTION_COUNT,
                           DEFAULT_GROUP_COMMIT_INTERVAL,
                           DEFAULT_OPS_COUNT,
                           DEFAULT_ABORT_MODE,
                           DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                           DEFAULT_GOETZ_MODE,
                           DEFAULT_REDO_LENGTH,
                           DEFAULT_REDO_FRACTION,
                           DEFAULT_STORAGE_RATIO)

            # COLLECT STATS
            collect_stats(TPCC_LATENCY_DIR, TPCC_LATENCY_CSV, TPCC_LATENCY_EXPERIMENT)

# NVM LATENCY -- EVAL
def nvm_latency_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(NVM_LATENCY_DIR)

    for nvm_latency in NVM_LATENCIES:

        # SET NVM LATENCY
        set_nvm_latency(nvm_latency)

        for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
            for nvm_logging_type in NVM_LOGGING_TYPES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               nvm_logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               nvm_latency,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

                # COLLECT STATS
                collect_stats(NVM_LATENCY_DIR, NVM_LATENCY_CSV, NVM_LATENCY_EXPERIMENT)

    # RESET NVM LATENCY
    reset_nvm_latency()

# PCOMMIT LATENCY -- EVAL
def pcommit_latency_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(PCOMMIT_LATENCY_DIR)

    nvm_wbl_logging_type = 1

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
            for pcommit_latency in PCOMMIT_LATENCIES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               nvm_wbl_logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               pcommit_latency,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

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
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               flush_mode,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

                # COLLECT STATS
                collect_stats(FLUSH_MODE_DIR, FLUSH_MODE_CSV, FLUSH_MODE_EXPERIMENT)


# ASYNCHRONOUS MODE -- EVAL
def asynchronous_mode_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(ASYNCHRONOUS_MODE_DIR)

    async_update_ratio = YCSB_UPDATE_RATIOS[2]

    for logging_type in LOGGING_TYPES:
        for asynchronous_mode in ASYNCHRONOUS_MODES:

            # RUN EXPERIMENT
            run_experiment(LOGGING,
                           EXPERIMENT_TYPE_THROUGHPUT,
                           logging_type,
                           YCSB_BENCHMARK_TYPE,
                           DEFAULT_CLIENT_COUNT,
                           DEFAULT_DURATION,
                           async_update_ratio,
                           DEFAULT_FLUSH_MODE,
                           INVALID_NVM_LATENCY,
                           INVALID_PCOMMIT_LATENCY,
                           asynchronous_mode,
                           INVALID_TRANSACTION_COUNT,
                           DEFAULT_GROUP_COMMIT_INTERVAL,
                           DEFAULT_OPS_COUNT,
                           DEFAULT_ABORT_MODE,
                           DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                           DEFAULT_GOETZ_MODE,
                           DEFAULT_REDO_LENGTH,
                           DEFAULT_REDO_FRACTION,
                           DEFAULT_STORAGE_RATIO)

            # COLLECT STATS
            collect_stats(ASYNCHRONOUS_MODE_DIR, ASYNCHRONOUS_MODE_CSV, ASYNCHRONOUS_MODE_EXPERIMENT)

# GROUP COMMIT -- EVAL
def group_commit_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(GROUP_COMMIT_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for logging_type in LOGGING_TYPES:
            for group_commit_interval in GROUP_COMMIT_INTERVALS:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               group_commit_interval,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

                # COLLECT STATS
                collect_stats(GROUP_COMMIT_DIR, GROUP_COMMIT_CSV, GROUP_COMMIT_EXPERIMENT)

# TIME TO COMMIT -- EVAL
def time_to_commit_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TIME_TO_COMMIT_DIR)

    ycsb_update_ratio = 1.0

    for logging_type in LOGGING_TYPES:
        for op_count in OPS_COUNT:
            for abort_mode in ABORT_MODES:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               op_count,
                               abort_mode,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,                              
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

                # COLLECT STATS
                collect_stats(TIME_TO_COMMIT_DIR, TIME_TO_COMMIT_CSV, TIME_TO_COMMIT_EXPERIMENT)

# LONG RUNNING TXN -- EVAL
def long_running_txn_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(LONG_RUNNING_TXN_DIR)

    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for logging_type in LOGGING_TYPES:
            for long_running_txn_count in LONG_RUNNING_TXN_COUNTS:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               DEFAULT_CLIENT_COUNT,
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               long_running_txn_count,
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               DEFAULT_STORAGE_RATIO)

                # COLLECT STATS
                collect_stats(LONG_RUNNING_TXN_DIR, LONG_RUNNING_TXN_CSV, LONG_RUNNING_TXN_EXPERIMENT)

# GOETZ -- EVAL
def goetz_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(GOETZ_DIR)

    goetz_mode = "1";
    
    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for logging_type in LOGGING_TYPES:
            for redo_length in REDO_LENGTHS:
                for redo_fraction in REDO_FRACTIONS:

                    # RUN EXPERIMENT
                    run_experiment(LOGGING,
                                   EXPERIMENT_TYPE_THROUGHPUT,
                                   logging_type,
                                   YCSB_BENCHMARK_TYPE,
                                   DEFAULT_CLIENT_COUNT,
                                   DEFAULT_DURATION,
                                   ycsb_update_ratio,
                                   DEFAULT_FLUSH_MODE,
                                   INVALID_NVM_LATENCY,
                                   INVALID_PCOMMIT_LATENCY,
                                   DEFAULT_ASYNCHRONOUS_MODE,
                                   INVALID_TRANSACTION_COUNT,
                                   DEFAULT_GROUP_COMMIT_INTERVAL,
                                   DEFAULT_OPS_COUNT,
                                   DEFAULT_ABORT_MODE,
                                   DEFAULT_LONG_RUNNING_TXN_COUNT,
                                   goetz_mode,
                                   redo_length,
                                   redo_fraction,
                                   DEFAULT_STORAGE_RATIO)
    
                    # COLLECT STATS
                    collect_stats(GOETZ_DIR, GOETZ_CSV, GOETZ_EXPERIMENT)

# HYBRID -- EVAL
def hybrid_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(HYBRID_DIR)

    logging_type = 1
    
    for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
        for client_count in CLIENT_COUNTS:
            for hybrid_storage_ratio in HYBRID_STORAGE_RATIOS:

                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_THROUGHPUT,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               client_count,
                               DEFAULT_DURATION,
                               ycsb_update_ratio,
                               DEFAULT_FLUSH_MODE,
                               INVALID_NVM_LATENCY,
                               INVALID_PCOMMIT_LATENCY,
                               DEFAULT_ASYNCHRONOUS_MODE,
                               INVALID_TRANSACTION_COUNT,
                               DEFAULT_GROUP_COMMIT_INTERVAL,
                               DEFAULT_OPS_COUNT,
                               DEFAULT_ABORT_MODE,
                               DEFAULT_LONG_RUNNING_TXN_COUNT,
                               DEFAULT_GOETZ_MODE,
                               DEFAULT_REDO_LENGTH,
                               DEFAULT_REDO_FRACTION,
                               hybrid_storage_ratio)

                # COLLECT STATS
                collect_stats(HYBRID_DIR, HYBRID_CSV, HYBRID_EXPERIMENT)


###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="log",
                                     description='Run Write Behind Logging Experiments',
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50))

    parser.add_argument("-z", "--enable-sdv", help='enable sdv', action='store_true')

    #parser.add_argument("-a", "--ycsb_throughput_eval", help='eval ycsb_throughput', action='store_true')
    #parser.add_argument("-b", "--tpcc_throughput_eval", help='eval tpcc_throughput', action='store_true')
    #parser.add_argument("-c", "--ycsb_recovery_eval", help='eval ycsb_recovery', action='store_true')
    #parser.add_argument("-d", "--tpcc_recovery_eval", help='eval tpcc_recovery', action='store_true')
    #parser.add_argument("-e", "--ycsb_latency_eval", help='eval ycsb_latency', action='store_true')
    #parser.add_argument("-f", "--tpcc_latency_eval", help='eval tpcc_latency', action='store_true')
    parser.add_argument("-g", "--nvm_latency_eval", help='eval nvm_latency', action='store_true')
    parser.add_argument("-i", "--pcommit_latency_eval", help='eval pcommit_latency', action='store_true')
    parser.add_argument("-j", "--flush_mode_eval", help='eval flush_mode', action='store_true')
    parser.add_argument("-k", "--asynchronous_mode_eval", help='eval asynchronous_mode', action='store_true')

    parser.add_argument("-a", "--group_commit_eval", help='eval group_commit', action='store_true')
    parser.add_argument("-b", "--time_to_commit_eval", help='eval time_to_commit', action='store_true')
    parser.add_argument("-c", "--long_running_txn_eval", help='eval long_running_txn', action='store_true')
    parser.add_argument("-d", "--goetz_eval", help='eval goetz', action='store_true')
    parser.add_argument("-e", "--hybrid_eval", help='eval hybrid', action='store_true')

    #parser.add_argument("-m", "--ycsb_throughput_plot", help='plot ycsb_throughput', action='store_true')
    #parser.add_argument("-n", "--tpcc_throughput_plot", help='plot tpcc_throughput', action='store_true')
    #parser.add_argument("-o", "--ycsb_recovery_plot", help='plot ycsb_recovery', action='store_true')
    #parser.add_argument("-p", "--tpcc_recovery_plot", help='plot tpcc_recovery', action='store_true')
    #parser.add_argument("-q", "--ycsb_storage_plot", help='plot ycsb_storage', action='store_true')
    #parser.add_argument("-r", "--tpcc_storage_plot", help='plot tpcc_storage', action='store_true')
    parser.add_argument("-s", "--ycsb_latency_plot", help='plot ycsb_latency', action='store_true')
    parser.add_argument("-t", "--tpcc_latency_plot", help='plot tpcc_latency', action='store_true')
    parser.add_argument("-u", "--nvm_latency_plot", help='plot nvm_latency', action='store_true')
    parser.add_argument("-v", "--pcommit_latency_plot", help='plot pcommit_latency', action='store_true')
    parser.add_argument("-w", "--flush_mode_plot", help='plot flush_mode', action='store_true')
    parser.add_argument("-x", "--asynchronous_mode_plot", help='plot asynchronous_mode', action='store_true')
    parser.add_argument("-y", "--replication_plot", help='plot replication', action='store_true')
    #parser.add_argument("-m", "--motivation_plot", help='plot motivation', action='store_true')

    parser.add_argument("-m", "--group_commit_plot", help='plot group commit', action='store_true')
    parser.add_argument("-n", "--time_to_commit_plot", help='eval time_to_commit', action='store_true')
    parser.add_argument("-o", "--long_running_txn_plot", help='eval long_running_txn', action='store_true')
    parser.add_argument("-p", "--goetz_plot", help='eval goetz', action='store_true')
    parser.add_argument("-q", "--hybrid_plot", help='eval hybrid', action='store_true')

    args = parser.parse_args()

    if args.enable_sdv:
        ENABLE_SDV = os.path.exists(SDV_DIR)
        print("ENABLE_SDV : " + str(ENABLE_SDV))

    ## EVAL

    #if args.ycsb_throughput_eval:
    #    ycsb_throughput_eval()

    #if args.tpcc_throughput_eval:
    #    tpcc_throughput_eval()

    #if args.ycsb_recovery_eval:
    #    ycsb_recovery_eval()

    #if args.tpcc_recovery_eval:
    #    tpcc_recovery_eval()

    #if args.ycsb_latency_eval:
    #    ycsb_latency_eval()

    #if args.tpcc_latency_eval:
    #    tpcc_latency_eval()

    if args.nvm_latency_eval:
        nvm_latency_eval()

    if args.pcommit_latency_eval:
        pcommit_latency_eval()

    if args.flush_mode_eval:
        flush_mode_eval()

    if args.asynchronous_mode_eval:
        asynchronous_mode_eval()

    if args.group_commit_eval:
        group_commit_eval()

    if args.time_to_commit_eval:
        time_to_commit_eval()

    if args.long_running_txn_eval:
        long_running_txn_eval()

    if args.goetz_eval:
        goetz_eval()

    if args.hybrid_eval:
        hybrid_eval()

    ## PLOT

    #if args.ycsb_throughput_plot:
    #    ycsb_throughput_plot()

    #if args.tpcc_throughput_plot:
    #    tpcc_throughput_plot()

    #if args.ycsb_recovery_plot:
    #    ycsb_recovery_plot()

    #if args.tpcc_recovery_plot:
    #    tpcc_recovery_plot()

    #if args.ycsb_storage_plot:
    #    ycsb_storage_plot()

    #if args.tpcc_storage_plot:
    #    tpcc_storage_plot()

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

    #if args.motivation_plot:
    #    motivation_plot()

    if args.replication_plot:
        replication_plot()

    if args.group_commit_plot:
        group_commit_plot()

    if args.time_to_commit_plot:
        time_to_commit_plot()

    if args.long_running_txn_plot:
        long_running_txn_plot()

    if args.goetz_plot:
        goetz_plot()

    if args.hybrid_plot:
        hybrid_plot()

    #create_legend_logging_types(False, False)
    #create_legend_logging_types(False, True)
    #create_legend_logging_types(True, False)
    #create_legend_logging_types(True, True)
    #create_legend_storage()
    #create_legend_update_ratio()
