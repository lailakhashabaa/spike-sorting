import statistics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def max_difference(samples):
    return max([abs(j - i) for i, j in zip(samples[:-1], samples[1:])])

def extract_spike_window(data, index, window_size=48):
    half_window = window_size // 2
    start = max(index - half_window, 0)
    end = min(index + half_window, len(data))
    spike_window = data[start:end]

    # Pad the spike window if necessary
    if len(spike_window) < window_size:
        if start == 0:
            # Pad at the beginning
            spike_window = [0] * (window_size - len(spike_window)) + spike_window
        else:
            # Pad at the end
            spike_window += [0] * (window_size - len(spike_window))

    return spike_window

def extract_spikes(data, threshold, window_size=48):
    spikes = []
    spikesIndex = []
    localMax = 0.0
    localMaxIndex = 0
    prevMax = False
    for i in range(len(data)):
        if data[i] > threshold:
            if data[i] > localMax:
                localMax = data[i]
                localMaxIndex = i
            prevMax = True
        elif prevMax:
            spikesIndex.append(localMaxIndex)
            spikes.append(localMax)
            localMax = 0.0
            prevMax = False
            i = localMaxIndex + (window_size//2)
            localMaxIndex = 0
    return spikes, spikesIndex

def calculate_spike_features(data, spikesIndex):
    std_devs = []
    max_diffs = []
    for i in range(len(spikesIndex)):
        start = spikesIndex[i] - 24 if i > 0 else 0
        end = spikesIndex[i] + 24
        spike_samples = data[start:end]
        std_devs.append(statistics.stdev(spike_samples))
        max_diffs.append(max_difference(spike_samples))
    return std_devs, max_diffs

def cluster_spikes(spikes, spikesIndex, std_devs, max_diffs):
    features = list(zip(std_devs, max_diffs))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    clusters = kmeans.labels_
    neuron1value = []
    neuron1index = []
    neuron2value = []
    neuron2index = []
    for i in range(len(clusters)):
        if clusters[i] == 0:
            neuron1value.append(spikes[i])
            neuron1index.append(spikesIndex[i])
        else:
            neuron2value.append(spikes[i])
            neuron2index.append(spikesIndex[i])

    plt.scatter(std_devs, max_diffs, c=clusters)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Max Difference")
    plt.show()
    return neuron1value, neuron1index, neuron2value, neuron2index

def calculate_mean_spike(data, neuronSpikeIndex):
    neuronWindow = []
    for i in neuronSpikeIndex:
        neuronWindow.append(extract_spike_window(data,i))
    
    return np.mean(neuronWindow, axis=0)


def spike_sorting(inputFile):
    # read file
    f = open(inputFile, "r") 
    lines = f.readlines()
    f.close()
    electrode1 = []
    electrode2 = []
    for line in lines:
        words = line.split()
        electrode1.append(float(words[0]))
        electrode2.append(float(words[1]))

    # extract spikes
    threshold1 = 3.5*statistics.stdev(electrode1[0:500])
    threshold2 = 3.5*statistics.stdev(electrode2[0:500])
    spikes1, spikes1index = extract_spikes(electrode1, threshold1)
    spikes2, spikes2index = extract_spikes(electrode2, threshold2)

    # Calculate standard deviation and max difference for each spike
    std_devs1, max_diffs1 = calculate_spike_features(electrode1, spikes1index)
    std_devs2, max_diffs2 = calculate_spike_features(electrode2, spikes2index)

    # Plotting the standard deviation of each spike vs the max difference of each spike
    plt.scatter(std_devs1, max_diffs1)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Max Difference")
    plt.show()

    # Plotting the standard deviation of each spike vs the max difference of each spike
    plt.scatter(std_devs2, max_diffs2)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Max Difference")
    plt.show()

    electrode1neuron1value, electrode1neuron1index, electrode1neuron2value, electrode1neuron2index = cluster_spikes(spikes1, spikes1index, std_devs1, max_diffs1)
    electrode2neuron1value, electrode2neuron1index, electrode2neuron2value, electrode2neuron2index = cluster_spikes(spikes2, spikes2index, std_devs2, max_diffs2)

    neuron1value = [electrode1neuron1value, electrode2neuron1value]
    neuron1index = [electrode1neuron1index, electrode2neuron1index]
    neuron2value = [electrode1neuron2value, electrode2neuron2value]
    neuron2index = [electrode1neuron2index, electrode2neuron2index]

    meanspike = [
        [calculate_mean_spike(electrode1, electrode1neuron1index), calculate_mean_spike(electrode1, electrode1neuron2index)],
        [calculate_mean_spike(electrode2, electrode2neuron1index), calculate_mean_spike(electrode2, electrode2neuron2index)]
    ]

    plt.plot(meanspike[0][0], color='blue')
    plt.plot(meanspike[0][1], color='red')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (uV)")
    plt.show()
    plt.plot(meanspike[1][0], color='blue')
    plt.plot(meanspike[1][1], color='red')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (uV)")
    plt.show()


    plt.figure(figsize=(15, 5))
    plt.plot(electrode1[:20000], label='Electrode 1 Raw Data')
    for i in range(len(neuron1index[0])):
        if neuron1index[0][i] < 20000:
            plt.plot(neuron1index[0][i], neuron1value[0][i], 'r*', label='Neuron 1' if i == 0 else "")
    for i in range(len(neuron2index[0])):
        if neuron2index[0][i] < 20000:
            plt.plot(neuron2index[0][i], neuron2value[0][i], 'b*', label='Neuron 2' if i == 0 else "")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Electrode 1 Spike Detection')
    plt.legend()
    plt.show()

    timestamps = [
        [[], []], # Electrode 1
        [[], []]  # Electrode 2
    ]
    for i in range(len(neuron1index[0])):
        timestamps[0][0].append(neuron1index[0][i] / 24414)
    for i in range(len(neuron2index[0])):
        timestamps[0][1].append(neuron2index[0][i] / 24414)
    for i in range(len(neuron1index[1])):
        timestamps[1][0].append(neuron1index[1][i] / 24414)
    for i in range(len(neuron2index[1])):
        timestamps[1][1].append(neuron2index[1][i] / 24414)

    return timestamps, meanspike

spike_sorting("Data.txt")