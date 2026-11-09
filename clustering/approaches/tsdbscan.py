from sktime.clustering.dbscan import TimeSeriesDBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def reduce_segments(segments, mode):
    if mode > 1:
        raise Exception('wrong mode value')
    reduced_segments = np.zeros([len(segments), len(segments[0]) - mode, len(segments[0][0])])

    for i in range(len(segments)):
        if mode == 0:
            reduced_segments[i] = segments[i]
        else:
            is_first = True
            seg = np.zeros( [len(segments[i])-1, len(segments[0][0])])
            for j in range(len(segments[i])):
                if is_first:
                    is_first = False
                else:
                    seg[j-1] = segments[i][j]
            reduced_segments[i] = seg
    return reduced_segments

def process_results(cluster_labels, labeled_segments, clusters):
    #heatmap
    labels = []
    highest_label = 0
    for seg in labeled_segments:
        labels.append(int(seg[0][0]))
        if seg[0][0] > highest_label:
            highest_label = int(seg[0][0])

    data = np.zeros((highest_label+1, clusters), dtype=float)
    for i in range(0, cluster_labels.size):
        data[int(labels[i])][cluster_labels[i]]+=1
    sns.heatmap(data)
    plt.title(f'k-means {clusters} heatmap')
    plt.xlabel('cluster labels')
    plt.ylabel('people')
    plt.savefig(f"results//kmeans_heatmap_{clusters}.png")
    plt.close()

def execute_ts_dbscan(raw_segments, mode="dtw"):
    segments = reduce_segments(raw_segments, 1)
    tsdbscan = TimeSeriesDBSCAN(
        distance=mode,
        eps=0.5,
        min_samples=2
    )
    tsdbscan.fit(segments)
    labels = tsdbscan.labels_
    process_results(labels, raw_segments, labels.max()+1)
