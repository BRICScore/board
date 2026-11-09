from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tslearn.metrics import dtw, dtw_path, cdist_dtw
import copy
from data_segmentator import DataSegmentator
import seaborn as sns


def clusters_evaluation(segments):
    print("to evaluate possible cluster sizes I will only use 10% of available data")
    reduced_segments = []
    for i in range(len(segments)):
        if i%10 == 0:
            seg = []
            is_first = True
            for s in segments[i]:
                if is_first:
                    is_first = False
                else:
                    seg.append(s)
            reduced_segments.append(seg)
    print("segment reduction completed, ", len(reduced_segments))
    sum_of_square_distances = []
    K = range(2, 14)
    for k in K:
        print("starting testing",k,"clusters")
        km = TimeSeriesKMeans(n_clusters=k,
                            n_init=2, 
                            metric="dtw", 
                            verbose=False, 
                            max_iter_barycenter=10,
                            random_state=0)
        km = km.fit(reduced_segments)
        sum_of_square_distances.append(km.inertia_)
    
    plt.plot(K, sum_of_square_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.savefig("results//reduced_kmeans_inertia.png")
    plt.show()

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

def calculate_derivative_dynamic_time_warping_distances(tensor_data, centroids, mode):
    simple_centroids = []
    for i in range(centroids.size()[0]):
        simple_centroids.append(get_middle_sensor_data(centroids[i], mode)[1])

    distances = np.zeros( [tensor_data.size()[0], centroids.size()[0]])
    for i in range(tensor_data.size()[0]):
        simple_data = get_middle_sensor_data(tensor_data[i], mode)[1]
        sim_data_derivative = seq_derivative(pd.DataFrame(simple_data))
        for j in range(len(simple_centroids)):
            sim_cent_derivative = seq_derivative(pd.DataFrame(simple_centroids[j]))
            distances[i][j] = dtw(sim_data_derivative, sim_cent_derivative)
    return torch.tensor(distances)

def calculate_dynamic_time_warping_distances(tensor_data, centroids, mode):
    simple_centroids = []
    for i in range(centroids.size()[0]):
        simple_centroids.append(get_middle_sensor_data(centroids[i], mode)[1])

    distances = np.zeros( [tensor_data.size()[0], centroids.size()[0]])
    for i in range(tensor_data.size()[0]):
        simple_data = get_middle_sensor_data(tensor_data[i], mode)[1]
        for j in range(len(simple_centroids)):
            distances[i][j] = dtw(simple_data, simple_centroids[j])
    
    return torch.tensor(distances)

def column_at_index(tensor, ind):
    res = []
    for t in tensor:
        res.append(t[ind])
    return res

def update_centroids(tensor_data, centroids, labels, mode):
    number_of_clusters = centroids.size()[0]
    new_centroids = np.zeros_like(centroids)

    for i in range(number_of_clusters):
        cluster = tensor_data[labels == i]
        test = torch.mean(cluster, dim=0)
        t = np.zeros_like(centroids)
        t[i] = test
        distances = calculate_derivative_dynamic_time_warping_distances(cluster, torch.tensor(t), mode)
        #distances = calculate_dynamic_time_warping_distances(cluster, torch.tensor(t), mode)
        dist = column_at_index(distances, i)
        t, ind = torch.min(torch.tensor(dist), dim=0)
        new_centroids[i] = cluster[ind]
    return torch.tensor(new_centroids)

def kmeans(raw_segments, number_of_clusters, mode):

    print("starting kmeans for", number_of_clusters,"clusters\n")
    segments = reduce_segments(raw_segments, mode)
    tensor_data = torch.tensor(segments)
    #plt.plot(segments[0])
    #r = np.random.randint(tensor_data.size()[0], size=10)
    #for i in r:
    #    mid = get_middle_sensor_data(tensor_data[i], mode)[1]
    #    print(mid)
    #    plt.plot(mid)
    plt.show()

    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:number_of_clusters]]
    test = torch.tensor(np.zeros([1]))
    max_iterations = 50

    print("start training model,", max_iterations, "iterations")
    for ind in range(max_iterations):

        distances = calculate_derivative_dynamic_time_warping_distances(tensor_data, centroids, mode)
        #distances = calculate_dynamic_time_warping_distances(tensor_data, centroids, mode)
        _, labels = torch.min(distances, dim=1)
        new_centroids = update_centroids(tensor_data, centroids, labels, mode)
        if torch.equal(new_centroids, centroids):
            print("training done after", ind, "iterations")
            break
        else:
            centroids = new_centroids
        #for i in range(number_of_clusters):
        #    if torch.sum(labels == i) > 0:
        #        centroids[i] = torch.mean(tensor_data[labels == i], dim=0)
    print("finished")  
    
    plt.figure()
    for i in range(number_of_clusters):
        plt.subplot(number_of_clusters, 1, i+1)
        for j in range(tensor_data.size()[0]):
            if labels[j] == i:
                plt.plot(get_middle_sensor_data(tensor_data[j], mode)[1], "k-", alpha=.2)
        plt.plot(get_middle_sensor_data(centroids[i], mode)[1], "r-")
    name = "results//kmeans_cluster_"+str(number_of_clusters)+".png"
    plt.savefig(name)
    plt.show()


    # presenting results
    if mode == 1:
        y = np.zeros_like(labels)
        for i in range(labels.size()[0]):
            y[i] = i
        plt.scatter(y, labels)
        plt.savefig("results//kmeans_clusters_distribution.png")
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
        plt.savefig("results//kmeans_clusters_distribution_tested.png")
        plt.show()

        print("distribution data")

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
        plt.savefig("results//kmeans_clusters_distribution_violin.png")
        plt.show()

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
    #plt.show()



def execute_kmeans(labeled_segments, clusters, mode="dtw"):
    segments = reduce_segments(labeled_segments, 1)
    if clusters != 0:
        labels, inertia = tslearn_kmeans(segments, clusters, mode)
        process_results(labels, labeled_segments, clusters)
    else:
        km_runs = 10
        labels = []
        inertias = []
        for i in range(2, km_runs):
            label, inertia = tslearn_kmeans(segments, i, mode)
            labels.append(label)
            inertias.append(inertia)
            process_results(label, labeled_segments, i)
        plt.plot(range(2, km_runs), inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.savefig(f"results//kmeans_inertia_2-{km_runs}.png")
        plt.show()

def tslearn_kmeans(segments, number_of_clusters, mode):
    print("starting processing", number_of_clusters, "clusters")
    km = TimeSeriesKMeans(n_clusters=number_of_clusters,
                          max_iter=100,
                          n_init=2,
                          metric=mode,
                          verbose=False,
                          max_iter_barycenter=100).fit(segments)
    labels = km.labels_
    inertia = km.inertia_
    return labels, inertia
