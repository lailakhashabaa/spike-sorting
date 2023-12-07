import statistics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def max_difference(samples):
    return max([abs(j - i) for i, j in zip(samples[:-1], samples[1:])])
def extract_spike_window(data, index, window_size=48):
    spike_window = []
    print(index)
    print(len(data))
    start = index - 24 if index > 24 else 0
    end = index + 24
    print(start, end)
    for i in range(start, end):
        spike_window.append(data[i])

    return spike_window
def read_data(filename):
    f = open(filename, "r")
    # process file line by line
    lines = f.readlines()
    # close the file
    f.close()

    # create a list to store the data
    electrode1 = []
    electrode2 = []

    # loop over lines
    for line in lines:
        # split line into words
        words = line.split()
        # convert words into integers
        electrode1.append(float(words[0]))
        electrode2.append(float(words[1]))

    return electrode1, electrode2

def detect_spikes(electrode, threshold):
    spikes = []
    spikesindex = []
    localMax = 0.0
    localMaxIndex = 0
    prevMax = False
    for i in range(len(electrode)):
        if electrode[i] > threshold:
            if electrode[i] > localMax:
                localMax = electrode[i]
                localMaxIndex = i
            prevMax = True
        elif prevMax:
            spikesindex.append(localMaxIndex)
            spikes.append(localMax)
            localMax = 0.0
            prevMax = False
    return spikesindex, spikes


def calculate_threshold(electrode, window_size=500):
    # Calculate the standard deviation of the first 500 samples
    std_dev = statistics.stdev(electrode[:window_size])

    # Calculate the threshold
    threshold = 3.5 * std_dev

    return threshold

def calculate_spike_features( spikes_index, electrode, spikes, window_size=48):
    std_devs = []
    max_diffs = []
    for i in range(len(spikes)):
        start = spikes_index[i] - 24 if i > 0 else 0
        end = spikes_index[i]
        spike_samples = electrode[start:end]
        std_devs.append(statistics.stdev(spike_samples))
        max_diffs.append(max_difference(spike_samples))
    return std_devs, max_diffs

def cluster_spike_features(std_devs, max_diffs, k=2):
    features = list(zip(std_devs, max_diffs))

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
    clusters = kmeans.labels_
    plt.scatter(std_devs, max_diffs, c=clusters)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Max Difference")
    plt.show()


    return clusters

def extract_spike_windows(electrode, spikes, window_size=48):
    spike_windows = []
    for i in spikes:
        spike_windows.append(extract_spike_window(electrode, i))
    return spike_windows

# Read data from file
electrode1, electrode2 = read_data("Data.txt")

# Calculate the threshold for each electrode
threshold1 = calculate_threshold(electrode1)
threshold2 = calculate_threshold(electrode2)

# Detect spikes for each electrode
spikes1index, spikes1 = detect_spikes(electrode1, threshold1)
spikes2index, spikes2 = detect_spikes(electrode2, threshold2)


# Calculate the standard deviation and max difference for each spike
std_devs1, max_diffs1 = calculate_spike_features(spikes1index, electrode1, spikes1)
std_devs2, max_diffs2 = calculate_spike_features(spikes2index, electrode2, spikes2)

# Cluster the spikes
clusters1 = cluster_spike_features(std_devs1, max_diffs1)
clusters2 = cluster_spike_features(std_devs2, max_diffs2)

# Extract spike windows
spike_windows1 = extract_spike_windows(electrode1, spikes1index)
spike_windows2 = extract_spike_windows(electrode2, spikes2index)

def separate_neurons(clusters,spikes):
    neuron1_value = []
    neuron2_value = []
    neuron1_index = []
    neuron2_index = []
    for i in range(len(clusters)):
        if clusters[i] == 0:
            neuron1_value.append(spikes[i])
            neuron1_index.append(i)
        else:
            neuron2_value.append(spikes[i])
            neuron2_index.append(i)


# Calculate the mean spike for each neuron
mean_spike1 = np.mean(spike_windows1, axis=0)
mean_spike2 = np.mean(spike_windows2, axis=0)

def plot_mean_spike(mean_spike, color='blue'):
    plt.plot(mean_spike, color=color)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (uV)")
    plt.show()



def get_timestamps(spikes_index, sampling_rate=24414):
    timestamps = []
    for i in spikes_index:
        timestamps.append(i / sampling_rate)
    return timestamps

def plot_spikes(electrode, spikes_index, spikes, sampling_rate=24414):
    # Assuming a sampling rate, convert index to time (in seconds)
    time = [i / sampling_rate for i in range(len(electrode))]  # Time for the first 20000 samples

    # Plotting the first 20,000 samples with detected spikes for Electrode 1
    plt.figure(figsize=(15, 5))
    plt.plot(time, electrode, label='Electrode 1 Raw Data')

    # Convert spike indices to time and plot
    for i in range(len(spikes_index)):
        spike_time = spikes_index[i] / sampling_rate
        plt.plot(spike_time, spikes[i], 'r*', label='Neuron 1' if i == 0 else "")

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.title('Electrode 1 Spike Detection with Time')
    plt.legend()
    plt.show()


# Plotting the mean spike of each neuron
plt.plot(mean_spike1[0][0])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (uV)")
plt.show()
plt.plot(mean_spike1[0][1])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (uV)")
plt.show()
plt.plot(meanspike1[1][0])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (uV)")
plt.show()
plt.plot(meanspike1[1][1])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (uV)")
plt.show()
#A figure showing the average spike of each neuron colored with different colors. 
#The x-axis is time and the y-axis is voltage.
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


# Plotting the first 20,000 samples with detected spikes for Electrode 1
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

# Assuming a sampling rate, convert index to time (in seconds)
sampling_rate = 24414  # Samples per second
time = [i / sampling_rate for i in range(20000)]  # Time for the first 20000 samples

# Plotting the first 20,000 samples with detected spikes for Electrode 1
plt.figure(figsize=(15, 5))
plt.plot(time, electrode1[:20000], label='Electrode 1 Raw Data')

# Convert spike indices to time and plot
for i in range(len(neuron1index[0])):
    if neuron1index[0][i] < 20000:
        spike_time = neuron1index[0][i] / sampling_rate
        plt.plot(spike_time, neuron1value[0][i], 'r*', label='Neuron 1' if i == 0 else "")
for i in range(len(neuron2index[0])):
    if neuron2index[0][i] < 20000:
        spike_time = neuron2index[0][i] / sampling_rate
        plt.plot(spike_time, neuron2value[0][i], 'b*', label='Neuron 2' if i == 0 else "")

plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.title('Electrode 1 Spike Detection with Time')
plt.legend()
plt.show()