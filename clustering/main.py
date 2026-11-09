from sympy.strategies.core import switch

from data_segmentator import DataSegmentator
from approaches.kmeans import clusters_evaluation, kmeans, execute_kmeans as km
from approaches.hierarchical import hierarchical
from approaches.dbscan import dbscan
from approaches.tsdbscan import execute_ts_dbscan
import sys
import argparse

sys.setrecursionlimit(1500)


CLEAN_FOLDER_PATH = 'measurements_dataset//clean'

def execute_dbscan(segments):
    eps = 0.45
    #eps = 0.2
    min_samples = 3
    #min_samples = 5
    test = input("do you want ddtw distance instead of dtw? y - yes n - no\n")
    if test[0] == 'y':
        dbscan(segments, mode=0, eps=eps, min_samples=min_samples, type=0)
    else:
        dbscan(segments, mode=0, eps=eps, min_samples=min_samples, type=1)

def execute_kmeans(segments, metadata, clusters):
    # test = input("do you want new cluster eval? y - yes n - no\n")
    # if test[0] == 'y':
    #     clusters_evaluation(segments)
    # cluster_count = input("how many clusters?\n")
    # try:
    #     CLUSTER_COUNT = int(cluster_count)
    # except:
    #     return
    if clusters !=0:
        kmeans(segments, clusters, mode=0)
        return
    for i in range(0,8):
        kmeans(segments, i, mode=0)


def execute_hierarchical(segments):
    hierarchical(segments, mode=0, action_type=0)

def execute_segmentation(segment_len):
    # SEGMENT_LENGTH = 1200
    # print("current segment length:", str(SEGMENT_LENGTH),"- 2 minutes")
    # test = input("do you want to input a new number? y - yes n - no\n")
    # if test[0] == 'y':
    #     segment_len = input("please state new segment lenght:\n")
    #     try:
    #         SEGMENT_LENGTH = int(segment_len)
    #     except:
    #         return None
    print("starting segmentation with length:", segment_len,"\n")
    segmentator = DataSegmentator(segment_len)
    segmentator.iterate_files()
    segments, metadata = segmentator.fixed_apply_segmentation()
    #segments, metadata = segmentator.apply_segmentation()
    return segments, metadata

def setup_argument_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--approach', choices=['all', 'kmeans', 'dbscan', 'hier'], required=True, help="Choose which clustering method should be used")
    parser.add_argument('-m', '--method', choices=['dtw', 'ddtw'], required=True, help="Choose 'distance' calculation algorithm")
    parser.add_argument('-s', '--segment_length', type=int, help="Length of a data segment")
    parser.add_argument('-k','--kmeans_clusters', type=int, help="Number of clusters for kmeans. Default value is 0 and means processing from 2 cluster to 7 clusters.")

    return parser.parse_args()


def main():
    segments, metadata = execute_segmentation(1200)
    while True:
        command = input("enter what now:\ns - segment data k - kmeans h - hierarchical clustering d - dbscan q - quit\n")
        if command[0] == 'q':
            break
        elif command[0] == 'k':
            execute_kmeans(segments, metadata, 2)
        elif command[0] == 's':
            segments, metadata = execute_segmentation()
        elif command[0] == 'h':
            execute_hierarchical(segments)
        elif command[0] == 'd':
            execute_dbscan(segments)

def print_labels(metadata):
    print('label to person mapping')

if __name__ == "__main__":
    args = setup_argument_parsing()

    segment_length = 1200 if not args.segment_length else args.segment_length
    segmentator = DataSegmentator(segment_length)
    segmentator.iterate_files()
    segments, metadata = segmentator.fixed_apply_segmentation()

    if args.approach == "all":
        pass
    elif args.approach == "kmeans":
        #execute_kmeans(segments, metadata, 0 if not args.kmeans else args.kmeans)
        km(segments, args.kmeans_clusters)
    elif args.approach == "dbscan":
        #execute_dbscan(segments)
        execute_ts_dbscan(segments)
    elif args.approach == "hier":
        execute_hierarchical(segments)
    print_labels(metadata)



print("done")
