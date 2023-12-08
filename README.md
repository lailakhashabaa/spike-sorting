# spike-sorting

This Python script performs spike sorting on neural data obtained from two electrodes. Spike sorting is the process of identifying and categorizing individual neuronal spikes from raw neural recordings.

## Prerequisites

Make sure you have the following libraries installed:

- `statistics`
- `matplotlib`
- `numpy`
- `scikit-learn`

You can install them using the following command:

```bash
pip install statistics matplotlib numpy scikit-learn
```

## Usage

```python
spike_sorting("Data.txt")
```

Replace `"Data.txt"` with the path to your input data file. The data file is expected to contain two columns corresponding to the readings from two electrodes.

## Code Overview

### Functions

1. **max_difference(samples)**
   - Calculates the maximum difference between consecutive elements in a list.

2. **extract_spike_window(data, index, window_size=48)**
   - Extracts a window of data centered around a given index.

3. **extract_spikes(data, threshold, window_size=48)**
   - Identifies spikes in the data based on a specified threshold.

4. **calculate_spike_features(data, spikesIndex)**
   - Calculates standard deviation and maximum difference for each identified spike.

5. **cluster_spikes(spikes, spikesIndex, std_devs, max_diffs)**
   - Performs clustering on spike features using k-means algorithm.

6. **calculate_mean_spike(data, neuronSpikeIndex)**
   - Calculates the mean spike waveform for a given set of spike indices.

7. **spike_sorting(inputFile)**
   - Main function that reads data from a file, extracts spikes, performs clustering, and visualizes the results.

### Workflow

1. Read data from the input file, assuming it contains two columns for two electrodes.

2. Identify spikes in each electrode's data based on a threshold.

3. Calculate standard deviation and maximum difference features for each spike.

4. Cluster spikes using the k-means algorithm.

5. Visualize the standard deviation vs. maximum difference for each electrode.

6. Plot mean spike waveforms for each identified cluster.

7. Display spike detection on raw data, highlighting different clusters.

8. Generate spike timestamps for further analysis.

