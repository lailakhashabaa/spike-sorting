import statistics
import matplotlib.pyplot as plt
import numpy as np

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


f = open("Data.txt", "r")
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
#calculate the standard deviation of the data of the first 500 samples of each electrode 

threshold1 = 3.5*statistics.stdev(electrode1[0:500])
threshold2 = 3.5*statistics.stdev(electrode2[0:500])

spikes1index = []
spikes2index = []
spikes1 = []
spikes2 = []
localMax1 = 0.0
for i in range(len(electrode1)):
    if electrode1[i] > threshold1:
        if electrode1[i] > localMax1:
            localMax1 = electrode1[i]
        else:
            spikes1index.append(i)
            spikes1.append(localMax1)
            localMax1 = 0.0
            i = i + 24
localMax2 = 0.0
for i in range(len(electrode2)):
    if electrode2[i] > threshold2:
        if electrode2[i] > localMax2:
            localMax2 = electrode2[i]
        else:
            spikes2index.append(i)
            spikes2.append(localMax2)
            localMax2 = 0.0
            i = i + 24
print(spikes1index[0:10])
print(spikes1[0:10])

# Calculate standard deviation and max difference for each spike
std_devs1 = []
max_diffs1 = []
for i in range(len(spikes1index)):
    start = spikes1index[i] - 24 if i > 0 else 0
    end = spikes1index[i]
    spike_samples = electrode1[start:end]
    std_devs1.append(statistics.stdev(spike_samples))
    max_diffs1.append(max_difference(spike_samples))

# Plotting the standard deviation of each spike vs the max difference of each spike
plt.scatter(std_devs1, max_diffs1)
plt.xlabel("Standard Deviation")
plt.ylabel("Max Difference")
plt.show()





# electrode 2
std_devs2 = []
max_diffs2 = []
for i in range(len(spikes2index)):
    start = spikes2index[i] - 24 if i > 0 else 0
    end = spikes2index[i]
    spike_samples = electrode2[start:end]
    std_devs2.append(statistics.stdev(spike_samples))
    max_diffs2.append(max_difference(spike_samples))

# Plotting the standard deviation of each spike vs the max difference of each spike
plt.scatter(std_devs2, max_diffs2)
plt.xlabel("Standard Deviation")
plt.ylabel("Max Difference")
plt.show()

from sklearn.cluster import KMeans

features1 = list(zip(std_devs1, max_diffs1))
features2 = list(zip(std_devs2, max_diffs2))

# Apply k-means clustering with k=2 for electrode 1
kmeans1 = KMeans(n_clusters=2, random_state=0).fit(features1)
clusters1 = kmeans1.labels_

# Apply k-means clustering with k=2 for electrode 2
kmeans2 = KMeans(n_clusters=2, random_state=0).fit(features2)
clusters2 = kmeans2.labels_

#plot the standard deviation of each spike vs the max difference of each spike for electrode 2 with k = 2
plt.scatter(std_devs2, max_diffs2, c=clusters2)
plt.xlabel("Standard Deviation")
plt.ylabel("Max Difference")
plt.show()

#plot the standard deviation of each spike vs the max difference of each spike for electrode 1 with k = 2
plt.scatter(std_devs1, max_diffs1, c=clusters1)
plt.xlabel("Standard Deviation")
plt.ylabel("Max Difference")
plt.show()


# seperate data of neruon 1 and neuron 2
neuron1value = [[], []]
neuron1index = [[], []]
neuron2value = [[], []]
neuron2index = [[], []]
for i in range(len(clusters1)):
    if clusters1[i] == 0:
        neuron1value[0].append(spikes1[i])
        neuron1index[0].append(spikes1index[i])
    else:
        neuron2value[0].append(spikes1[i])
        neuron2index[0].append(spikes1index[i])
for i in range(len(clusters2)):
    if clusters2[i] == 0:
        neuron1value[1].append(spikes2[i])
        neuron1index[1].append(spikes2index[i])
    else:
        neuron2value[1].append(spikes2[i])
        neuron2index[1].append(spikes2index[i])

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

meanspike = [[], []]
# Extracting and averaging spike windows
neuron1windows = [[], []]
neuron2windows = [[], []]
for i in neuron1index[0]:
    neuron1windows[0].append(extract_spike_window(electrode1, i))
for i in neuron2index[0]:
    neuron2windows[0].append(extract_spike_window(electrode1, i))
for i in neuron1index[1]:
    neuron1windows[1].append(extract_spike_window(electrode2, i))
for i in neuron2index[1]:
    neuron2windows[1].append(extract_spike_window(electrode2, i))

# Calculating mean spikes
meanspike = [
    [np.mean(neuron1windows[0], axis=0), np.mean(neuron2windows[0], axis=0)],
    [np.mean(neuron1windows[1], axis=0), np.mean(neuron2windows[1], axis=0)]
]

# Print the mean spike of each neuron
print("Mean Spike of Neuron 1 in Electrode 1:", meanspike[0][0])
print("Mean Spike of Neuron 2 in Electrode 1:", meanspike[0][1])
print("Mean Spike of Neuron 1 in Electrode 2:", meanspike[1][0])
print("Mean Spike of Neuron 2 in Electrode 2:", meanspike[1][1])

# Plotting the mean spike of each neuron
plt.plot(meanspike[0][0])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (uV)")
plt.show()
plt.plot(meanspike[0][1])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (uV)")
plt.show()
plt.plot(meanspike[1][0])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (uV)")
plt.show()
plt.plot(meanspike[1][1])
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