import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tslearn.metrics import dtw, dtw_path, cdist_dtw

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

def ddtw_distance(tensor_a, tensor_b, mode):
    a = seq_derivative(pd.DataFrame(get_middle_sensor_data(tensor_a, mode)[1]))
    b = seq_derivative(pd.DataFrame(get_middle_sensor_data(tensor_b, mode)[1]))

    return dtw(a, b)

def dtw_distance(tensor_a, tensor_b, mode):
    a = get_middle_sensor_data(tensor_a, mode)[1]
    b = get_middle_sensor_data(tensor_b, mode)[1]

    return dtw(a, b)

def expand_cluster(tensor_data, labels, visited, neighbours, cluster_label, eps, min_samples, mode, type):
    i = 0
    while i < neighbours.shape[0]:
        neighbour_ind = neighbours[i].item()
        if not visited[neighbour_ind]:
            visited[neighbour_ind] = True
            neighbour_neighbours = torch.tensor(get_neighbours(tensor_data[neighbour_ind], tensor_data, mode, eps, tensor_data.shape[0]))
            if neighbour_neighbours.shape[0] >= min_samples:
                neighbours = torch.cat((neighbours, neighbour_neighbours))
            if labels[neighbour_ind] == 0:
                labels[neighbour_ind] = cluster_label
        i += 1

def get_neighbours(data_point, tensor_data, mode, eps, n_samples):
    n = []
    for j in range(n_samples):
        dist = 0
        if type == 0:
            dist = ddtw_distance(data_point, tensor_data[j], mode)
        else:
            dist = dtw_distance(data_point, tensor_data[j], mode)
        if dist < eps:
            n.append(j)

    return n

def dbscan(raw_segments, mode, eps, min_samples, type=0):
    segments = reduce_segments(raw_segments, mode)
    tensor_data = torch.tensor(segments)
    n_samples = tensor_data.shape[0]
    labels = torch.zeros(n_samples, dtype=torch.int)

    cluster_label = 0
    visited = torch.zeros(n_samples, dtype=torch.bool)

    eps_test = []

    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True

        neighbours = torch.tensor(get_neighbours(tensor_data[i], tensor_data, mode, eps, n_samples))
        eps_test.append(neighbours.shape[0])

        if neighbours.shape[0] < min_samples:
            # noise
            labels[i] = 0
        else:
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster(tensor_data, labels, visited, neighbours, cluster_label, eps, min_samples, mode, type)


    number_of_clusters = torch.unique(labels).shape[0] - 1
    plt.figure()
    for i in range(number_of_clusters):
        plt.subplot(number_of_clusters, 1, i + 1)
        for j in range(tensor_data.size()[0]):
            if labels[j] == i:
                plt.plot(get_middle_sensor_data(tensor_data[j], mode)[1], "k-", alpha=.2)
    if type == 0:
        method = 'ddtw_'
    else:
        method = 'dtw_'
    plt.savefig(f'results//${method}dbscan_clusters.png')
    plt.show()

    if mode == 1:
        y = np.zeros_like(labels)
        for i in range(labels.size()[0]):
            y[i] = i
        plt.scatter(y, labels)
        plt.savefig(f'results//${method}dbscan_clusters_distribution.png')
        plt.show()
    else:
        # we have data to test against :c
        tested_labels = []
        tested_y = []
        # 14 because there are 14 files in data set, it will propably be too much
        for i in range(14):
            tested_labels.append([])
            tested_y.append([])

        for i in range(tensor_data.size()[0]):
            idn = int(tensor_data[i][0][0])
            tested_labels[idn].append(labels[i])
            tested_y[idn].append(i)

        for i in range(14):
            plt.scatter(tested_y[i], tested_labels[i])
        plt.savefig(f'results//${method}dbscan_clusters_distribution_tested.png')
        plt.show()

        violin_data = []
        for i in range(len(tested_labels)):
            labs = torch.tensor(tested_labels[i])
            if len(labs)>0:
                print("person",i,"data:")
                res = torch.bincount(labs)
                v = labs
                res_percent = []
                for r in res:
                    res_percent.append(float(r/len(labs)))
                print(res)
                print(res_percent)
                #violin_data.append(res)
                violin_data.append(v)

        fig = plt.figure()
        ax = fig.gca()
        ax.violinplot(violin_data)
        plt.savefig(f'results//${method}dbscan_clusters_distribution_violin.png')
        plt.show()