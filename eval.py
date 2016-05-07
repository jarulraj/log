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
NVM_LATENCIES = ("160", "320")
DEFAULT_NVM_LATENCY = NVM_LATENCIES[0]
ENABLE_SDV = False

OUTPUT_FILE = "outputfile.summary"

# Refer LoggingType in common/types.h
LOGGING_TYPES = (0, 1, 2, 3, 4)
RECOVERY_LOGGING_TYPES = LOGGING_TYPES[1:]

LOGGING_NAMES = ("disabled", 
                 "nvm_wal", "nvm_wbl", 
                 "hdd_wal", "hdd_wbl")
RECOVERY_LOGGING_NAMES = LOGGING_NAMES[1:]

SCALE_FACTOR = 1
DATABASE_FILE_SIZE = 4096  # DATABASE FILE SIZE (MB)

TRANSACTION_COUNT = 10

CLIENT_COUNTS = (1, 2, 4, 8)

YCSB_BENCHMARK_TYPE = 1
TPCC_BENCHMARK_TYPE = 2

RECOVERY_TRANSACTION_COUNTS = (1, 3, 5)

YCSB_UPDATE_RATIOS = (0, 0.1, 0.5, 0.9)
YCSB_UPDATE_NAMES = ("read-only", "read-heavy", "balanced", "write-heavy")

YCSB_SKEW_FACTORS = (1, 2)
YCSB_SKEW_NAMES = ("low-skew", "high-skew")

INVALID_UPDATE_RATIO = 0
INVALID_SKEW_FACTOR = 1;

EXPERIMENT_TYPE_THROUGHPUT = 1
EXPERIMENT_TYPE_RECOVERY = 2

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

# Figure out ycsb skew name
def getYCSBSkewName(ycsb_skew):
    
    ycsb_skew_offset = YCSB_SKEW_FACTORS.index(int(ycsb_skew))
    ycsb_skew_name = YCSB_SKEW_NAMES[ycsb_skew_offset]

    return ycsb_skew_name

###################################################################################
# PLOT
###################################################################################

def create_legend_logging_types():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(12, 0.5))

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
    x_labels = [str(i) for i in RECOVERY_TRANSACTION_COUNTS]
    N = len(x_labels)
    M = len(RECOVERY_LOGGING_NAMES)
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

###################################################################################
# PLOT HELPERS
###################################################################################

# THROUGHPUT -- PLOT
def ycsb_throughput_plot():

    for ycsb_skew_factor in YCSB_SKEW_FACTORS:

        ycsb_skew_name = getYCSBSkewName(ycsb_skew_factor)

        for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
    
            ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)
    
            datasets = []
            for logging_type in LOGGING_TYPES:
    
                # figure out logging name and ycsb update name
                logging_name = getLoggingName(logging_type)
        
                data_file = YCSB_THROUGHPUT_DIR + "/" + ycsb_skew_name + "/" + ycsb_update_name + "/" + logging_name + "/" + YCSB_THROUGHPUT_CSV
        
                dataset = loadDataFile(len(CLIENT_COUNTS), 2, data_file)
                datasets.append(dataset)
    
            fig = create_ycsb_throughput_line_chart(datasets)
        
            fileName = "ycsb-" + "throughput-" + ycsb_skew_name + "-" + ycsb_update_name + ".pdf"
        
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
    for logging_type in RECOVERY_LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = YCSB_RECOVERY_DIR + "/" + logging_name + "/" + YCSB_RECOVERY_CSV

        dataset = loadDataFile(len(RECOVERY_TRANSACTION_COUNTS), 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_recovery_bar_chart(datasets)

    fileName = "ycsb-" + "recovery" + ".pdf"
    
    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)
    
# TPCC RECOVERY -- PLOT
def tpcc_recovery_plot():

    datasets = []
    for logging_type in RECOVERY_LOGGING_TYPES:

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        data_file = TPCC_RECOVERY_DIR + "/" + logging_name + "/" + TPCC_RECOVERY_CSV

        dataset = loadDataFile(len(RECOVERY_TRANSACTION_COUNTS), 2, data_file)
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
                   transaction_count,
                   ycsb_update_ratio,
                   ycsb_skew_factor):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)
    
    subprocess.call([program,
                     "-e", str(experiment_type),
                     "-l", str(logging_type),
                     "-k", str(SCALE_FACTOR),
                     "-f", str(DATABASE_FILE_SIZE),
                     "-t", str(transaction_count),
                     "-b", str(client_count),
                     "-u", str(ycsb_update_ratio),
                     "-s", str(ycsb_skew_factor),
                     "-y", str(benchmark_type)])


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
        ycsb_skew_factor = data[5]
        transaction_count = data[6]

        stat = data[7]

        # figure out logging name and ycsb update name
        logging_name = getLoggingName(logging_type)

        # MAKE RESULTS FILE DIR
        if category == YCSB_THROUGHPUT_EXPERIMENT:
            ycsb_update_name = getYCSBUpdateName(ycsb_update_ratio)
            ycsb_skew_name = getYCSBSkewName(ycsb_skew_factor)
            result_directory = result_dir + "/" + ycsb_skew_name + "/" + ycsb_update_name + "/" + logging_name
        elif category == TPCC_THROUGHPUT_EXPERIMENT:
            result_directory = result_dir + "/" + logging_name
        elif category == YCSB_RECOVERY_EXPERIMENT or category == TPCC_RECOVERY_EXPERIMENT:
            result_directory = result_dir + "/" + logging_name

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")

        # WRITE OUT STATS
        if category == YCSB_THROUGHPUT_EXPERIMENT or category == TPCC_THROUGHPUT_EXPERIMENT:
            result_file.write(str(backend_count) + " , " + str(stat) + "\n")
        elif category == YCSB_RECOVERY_EXPERIMENT or category == TPCC_RECOVERY_EXPERIMENT:
            result_file.write(str(transaction_count) + " , " + str(stat) + "\n")

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

    for ycsb_skew_factor in YCSB_SKEW_FACTORS:
        for ycsb_update_ratio in YCSB_UPDATE_RATIOS:
            for logging_type in LOGGING_TYPES:
                for client_count in CLIENT_COUNTS:
                    
                    # RUN EXPERIMENT
                    run_experiment(LOGGING,
                                   EXPERIMENT_TYPE_THROUGHPUT,
                                   logging_type,
                                   YCSB_BENCHMARK_TYPE,
                                   client_count,
                                   TRANSACTION_COUNT,
                                   ycsb_update_ratio,
                                   ycsb_skew_factor)
    
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
                           TRANSACTION_COUNT,
                           INVALID_UPDATE_RATIO,
                           INVALID_SKEW_FACTOR)

            # COLLECT STATS
            collect_stats(TPCC_THROUGHPUT_DIR, TPCC_THROUGHPUT_CSV, TPCC_THROUGHPUT_EXPERIMENT)

# YCSB RECOVERY -- EVAL
def ycsb_recovery_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(YCSB_RECOVERY_DIR)

    client_count = 1
    ycsb_recovery_update_ratio = 1
    ycsb_recovery_skew_factor = 1
    
    for recovery_transaction_count in RECOVERY_TRANSACTION_COUNTS:
            for logging_type in LOGGING_TYPES:
                    
                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_RECOVERY,
                               logging_type,
                               YCSB_BENCHMARK_TYPE,
                               client_count,
                               recovery_transaction_count,
                               ycsb_recovery_update_ratio,
                               ycsb_recovery_skew_factor)

                # COLLECT STATS
                collect_stats(YCSB_RECOVERY_DIR, YCSB_RECOVERY_CSV, YCSB_RECOVERY_EXPERIMENT)

# TPCC RECOVERY -- EVAL
def tpcc_recovery_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TPCC_RECOVERY_DIR)

    client_count = 1
    
    for recovery_transaction_count in RECOVERY_TRANSACTION_COUNTS:
            for logging_type in LOGGING_TYPES:
                    
                # RUN EXPERIMENT
                run_experiment(LOGGING,
                               EXPERIMENT_TYPE_RECOVERY,
                               logging_type,
                               TPCC_BENCHMARK_TYPE,
                               client_count,
                               recovery_transaction_count,
                               INVALID_UPDATE_RATIO,
                               INVALID_SKEW_FACTOR)

                # COLLECT STATS
                collect_stats(TPCC_RECOVERY_DIR, TPCC_RECOVERY_CSV, TPCC_RECOVERY_EXPERIMENT)

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="log", 
                                     description='Run Write Behind Logging Experiments',
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50))

    parser.add_argument("-x", "--enable-sdv", help='enable sdv', action='store_true')
    parser.add_argument("-a", "--ycsb_throughput_eval", help='eval ycsb_throughput', action='store_true')
    parser.add_argument("-b", "--tpcc_throughput_eval", help='eval tpcc_throughput', action='store_true')
    parser.add_argument("-c", "--ycsb_recovery_eval", help='eval ycsb_recovery', action='store_true')
    parser.add_argument("-d", "--tpcc_recovery_eval", help='eval tpcc_recovery', action='store_true')

    parser.add_argument("-m", "--ycsb_throughput_plot", help='plot ycsb_throughput', action='store_true')
    parser.add_argument("-n", "--tpcc_throughput_plot", help='plot tpcc_throughput', action='store_true')
    parser.add_argument("-o", "--ycsb_recovery_plot", help='plot ycsb_recovery', action='store_true')
    parser.add_argument("-p", "--tpcc_recovery_plot", help='plot tpcc_recovery', action='store_true')
    parser.add_argument("-q", "--ycsb_storage_plot", help='plot ycsb_storage', action='store_true')
    parser.add_argument("-r", "--tpcc_storage_plot", help='plot tpcc_storage', action='store_true')

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

    #create_legend_logging_types()
    create_legend_storage()
