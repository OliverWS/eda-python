from qLogFile import *
import numpy as np
from affdex import utils as aff

def aggregate(qLogs, targetSamples):
    metrics = getMetricsForQLogs(qLogs)
    meanCurve, stdCurve, numResponse,  responseCoverage, sessions = aff.aggregateMetrics(metrics, "Q", targetSamples)
    return meanCurve
def getMetricsForQLogs(qLogs):
    metrics = []
    for q in qLogs:
        metric = aff.QMetricData(q)
        metrics.append(metric)
    return metrics