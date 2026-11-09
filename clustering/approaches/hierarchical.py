import torch
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.metrics import dtw, dtw_path, cdist_dtw


def get_middle_sensor_data(segment, mode):
    start = (mode + 1)%2
    result = np.zeros( [2, segment.size()[0]-start])
    for i in range(start, segment.size()[0]):
        result[0][i-start] = segment[i][0]
        result[1][i-start] = segment[i][3]
    return result

def seq_derivative(g):
    '''
    g: Pandas DataFrame
    '''
    prev = g.shift()
    next = g.shift(-1)
    dif = g.diff()
    rl_neigh = next - prev
    derivative = (dif + (rl_neigh/2))/2
    derivative.dropna(inplace=True)
    return derivative.to_numpy()

def reduce_segments(segments, mode):
    if mode > 1:
        raise Exception('wrong mode value')
    reduced_segments = np.zeros([len(segments), len(segments[0]) - mode, len(segments[0][0])])

    for i in range(len(segments)):
        if(mode == 0):
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

def ddtw_distance(tensor_a, tensor_b, mode):
    a = seq_derivative(pd.DataFrame(get_middle_sensor_data(tensor_a, mode)[1]))
    b = seq_derivative(pd.DataFrame(get_middle_sensor_data(tensor_b, mode)[1]))

    return dtw(a, b)

def distance_matrix_using_ddtw(tensor_data, mode):
    print(tensor_data.size())
    distances = np.zeros((tensor_data.size()[0], tensor_data.size()[0]))
    for i in range(tensor_data.size()[0]):
        for j in range(tensor_data.size()[0]):
            distances[i][j] = ddtw_distance(tensor_data[i], tensor_data[j], mode)
    
    return distances

def hierarchical(segments, mode, action_type):
    if action_type == 0:
        hierarchical_agglomerative(segments, mode)

def hierarchical_agglomerative(raw_segments, mode):
    print("starting agglomerative hierarchical clustering")
    segments = reduce_segments(raw_segments, mode)
    tensor_data = torch.tensor(segments)

    plt.plot(segments[0])
    plt.show()

    distances = distance_matrix_using_ddtw(tensor_data, mode)
    print("distances calculated")

    Z = linkage(distances, 'single')
    #print(Z)

    plt.figure(figsize=(10, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(Z)
    plt.savefig("results//hierarchical_a_dendrogram.png")
    plt.show()
    
