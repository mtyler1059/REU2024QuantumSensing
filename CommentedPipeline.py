import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from hdbscan import HDBSCAN
import random
import warnings
from collections import Counter
import pandas as pd

# Suppress all warnings
warnings.filterwarnings("ignore")

#Global variable what we're doing with IR Spectra
#1 = import variance data below, 2 = naive variances, 3 = pairwise differences
IR_Spectra = 2

# Global variable to select comb placement type
# KDE Integral: 1, Random: 2, KDE Npeaks: 3, Pairwise Differences = 4
# 1D array         1D array   1D array       2D array
comb_placement_type = 3

# Global variable for dimensionality reduction method
# PCA: 1, t-SNE: 2, SpectralClustering: 3, HDBSCAN: 4
dimensionality_reduction_method = 3

# Load comb data and IR Spectra data
# CombData: (m, n), where m is the number of combs + 1, and n is the number of frequencies
# IRSpectra: (p, q), where p is the number of molecules and q is the number of frequencies
# IRUnprocessed: (r, s), where r is the number of samples and s is the number of frequencies
CombData = np.abs(np.load('smoothed_shifted_combs_Var37_711.npy'))
IRSpectra = np.abs(np.load('10_IR_Spectra_date_7_5.npy'))
IRUnprocessed = np.abs(np.load('_7_15 IR Data nonCombed.npy'))

def find_variance(IRSpectra):
    """
    Calculate the variance of IR spectra across samples.

    Parameters:
    IRSpectra: (p, q)

    Returns:
    variance: (q,)
    """
    return np.var(IRSpectra, axis=0)

def pairwise_diff(IRSpectra):
    """
    Calculate the pairwise differences of IR spectra.

    Parameters:
    IRSpectra: (p, q)

    Returns:
    pairwise_diffs: (p*(p-1)/2, q)
    """
    num_rows, num_cols = IRSpectra.shape
    num_pairs = num_rows * (num_rows - 1) // 2
    pairwise_diffs = np.zeros((num_pairs, num_cols))

    index = 0
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            pairwise_diffs[index] = np.abs(IRSpectra[i] - IRSpectra[j])
            index += 1

    return pairwise_diffs

# Determine variance data based on IR_Spectra variable
if IR_Spectra == 1:
    # KDEVariance: (m, n) from loaded data
    KDEVariance = np.abs(np.load('PairsOf10IRSpectraSDs.npy'))[1:, :]
elif IR_Spectra == 2:
    # KDEVariance: (q,) from variance calculation
    KDEVariance = find_variance(IRSpectra[1:, :])
elif IR_Spectra == 3:
    # KDEVariance: (p*(p-1)/2, q) from pairwise differences
    KDEVariance = pairwise_diff(IRSpectra[1:, :])

# Frequencies axis for the KDE variance data
# frequencies: (n,)
frequencies = np.abs(np.load('xAxis.npy'))

# Parse the CombData
# numMolecules: scalar, number of molecules
# numCombs: scalar, number of combs
# combFrequencies: (n,)
# data: (m-1, n)
numMolecules = 10
numCombs = 4
combFrequencies = CombData[0, :]
data = CombData[1:, :]

# combLength: scalar, length of each comb in terms of frequency bins
totalLength = len(combFrequencies)
teeth_spacing = 1
freqGap = combFrequencies[1] - combFrequencies[0]
combLength = totalLength // 20
originalCombLength = combLength
combLength = (np.ceil(freqGap * combLength)).astype(int)

def max_integral(variances, length):
    """
    Find the starting index that maximizes the sum of variances over a given length.

    Parameters:
    variances: (q,)
    length: scalar

    Returns:
    startIndex: list of indices
    """
    startIndex = []
    max_sum = sum(variances[:length])
    max_index = 0
    current_sum = max_sum

    for i in range(1, len(variances) - length + 1):
        current_sum = current_sum - variances[i - 1] + variances[i + length - 1]
        if current_sum > max_sum:
            max_sum = current_sum
            max_index = i

    startIndex.append(max_index)
    return startIndex

def random_integral(variances, length):
    """
    Find a random starting index for variances over a given length.

    Parameters:
    variances: (q,)
    length: scalar

    Returns:
    startIndex: list of indices
    """
    startIndex = []
    startIndex.append(random.randint(0, len(variances) - length))
    return startIndex

def pairwise_differences(variances, length):
    """
    Find starting indices that maximize the sum of pairwise variances over a given length.
    (One comb per pair, this can lead to high requirements on combs for small molecule increases)

    Parameters:
    variances: (p*(p-1)/2, q)
    length: scalar

    Returns:
    startIndices: list of indices
    """
    startIndices = []
    for v in variances:
        startIndices.append(max_integral(v, length)[0])
    return startIndices

def partial_array_sum(array, start_index, end_index):
    """
    Calculate the sum of a portion of an array. Helper Method for npeaks.

    Parameters:
    array: (n,)
    start_index: scalar
    end_index: scalar

    Returns:
    total: scalar
    """
    total = 0
    for i in range(start_index, end_index):
        total += array[i-1]
    return total

def reduce_array(array, start_index, snip_length):
    """
    Reduce an array by setting a portion to -10000. Helper method for npeaks
    (removing the area one peak is taken at)

    Parameters:
    array: (n,)
    start_index: scalar
    snip_length: scalar

    Returns:
    new_array: (n,)
    """
    new_list = []
    for i in range(0, len(array)):
        if not start_index <= i < start_index + snip_length:
            new_list.append(array[i])
        else:
            new_list.append(-10000)
    return np.array(new_list)

def maximize_sum(values, length):
    """
    Find the starting index that maximizes the sum of values over a given length.
    Helper method for npeaks, similar to the single peak method above.

    Parameters:
    values: (q,)
    length: scalar

    Returns:
    start_val: scalar
    """
    max_val = 0
    start_val = 0
    for i in range(0, len(values)-length):
        if partial_array_sum(values, i, i+length) > max_val:
            max_val = partial_array_sum(values, i, i+length)
            start_val = i
    return start_val

def find_npeaks(array, length, n):
    """
    Find the starting indices of n peaks in an array.
    This method is for n disjoint combs.

    Parameters:
    array: (q,)
    length: scalar
    n: scalar

    Returns:
    npeaks: list of indices
    """
    npeaks = []
    array_m = array

    for i in range(0, n):
        start_index = maximize_sum(array_m, length)
        npeaks.append(start_index-1)
        array_m = reduce_array(array_m, start_index, length)

    return npeaks

def Joint_Method(var1d, varPairwise, combLength, n, threshold):
    """
    Joint method for finding peaks in 1D and pairwise variances.

    Parameters:
    var1d: (q,)
    varPairwise: (p*(p-1)/2, q)
    combLength: scalar
    n: scalar
    threshold: scalar

    Returns:
    startIndex: list of indices
    """
    startIndex = find_npeaks(var1d, combLength, n)

# Determine the start indices of combs based on placement type.
'''FOR ALL OF THESE, startIndex is the set of start indices for the combs, and their
length and other parameters are defined at the top. The above is for comb placement,
and the below is for passing through the comb and analyzing data.'''
if comb_placement_type == 1:
    startIndex = max_integral(KDEVariance, combLength)
elif comb_placement_type == 2:
    startIndex = random_integral(KDEVariance, combLength)
elif comb_placement_type == 3:
    startIndex = find_npeaks(KDEVariance, combLength, numCombs)
elif comb_placement_type == 4:
    startIndex = pairwise_differences(KDEVariance, combLength)
else:
    raise ValueError("Invalid comb placement type. Use 1 for KDE max index or 2 for random placement.")

# Starting here are functions for generation of the smaller combs.
#I didn't write these, but it largely feeds off Matty's code but parametrized.
def comb_x(cps1):
    """
    Generate comb x-values based on parameters.

    Parameters:
    cps1: dictionary of parameters

    Returns:
    comb_x_values: (n,)
    """
    comb_x_values = np.fft.fftfreq(n = int(cps1['time'] * cps1['sample_rate']), d = 1 / cps1['sample_rate'])
    return comb_x_values

def calculate_h(cps2, comb_x_values_i):
    """
    Calculate the H parameter for a comb.

    Parameters:
    cps2: dictionary of parameters
    comb_x_values_i: (n,)

    Returns:
    H_value: (n,)
    """
    path_length = 100e-3
    speed_of_light = 3e8
    refractive_index = cps2['n_0']
    absorption_coefficient = cps2['alpha_0']
    refractive_index_transformed = refractive_index + 0.1 * np.sin(comb_x_values_i*2*np.pi)
    absorption_coeffient_transoformed = absorption_coefficient * np.exp(-comb_x_values_i / 1.5e14)
    H_absorption_value = np.exp(-absorption_coeffient_transoformed * path_length)
    H_phase_value = np.exp(-1j * 2 * np.pi * comb_x_values_i * (refractive_index_transformed - 1) * path_length / speed_of_light)
    H_value = H_absorption_value * H_phase_value
    return H_value

def comb_y(cps3, comb_x_values_j):
    """
    Generate comb y-values based on parameters.

    Parameters:
    cps3: dictionary of parameters
    comb_x_values_j: (n,)

    Returns:
    final_amplitudes: (n,)
    """
    number_of_samples = int(cps3['time'] * cps3['sample_rate'])
    sample_set = np.zeros(number_of_samples)

    number_of_pulses_without_reference_to_samples = int(cps3['time'] * cps3['rep_rate'])
    amount_of_samples_coincident_with_pulses = int(cps3['pulse_duration'] * cps3['sample_rate'])

    pulse_drift_black_box = np.linspace(0,
                                      cps3['drift'] / cps3['rep_rate'],
                                      number_of_pulses_without_reference_to_samples) * np.exp(np.linspace(0,
                                                                                                          100 * cps3['drift'],
                                                                                                          number_of_pulses_without_reference_to_samples))
    pulse_times_noise_black_box = np.random.normal(loc = np.arange(number_of_pulses_without_reference_to_samples) / cps3['rep_rate'],
                                                 scale = cps3['jitter'] / cps3['rep_rate'],
                                                 size = number_of_pulses_without_reference_to_samples)

    actual_pulse_time_start_points = np.add(pulse_times_noise_black_box,
                                          pulse_drift_black_box)

    for actual_pulse_time_start_point in actual_pulse_time_start_points:
        starting_sample = int(actual_pulse_time_start_point * cps3['sample_rate'])
        if starting_sample + amount_of_samples_coincident_with_pulses < number_of_samples:
            sample_set[starting_sample:starting_sample + amount_of_samples_coincident_with_pulses] = 1

    sample_set += cps3['noise'] * np.random.normal(size = number_of_samples)

    fourier_amplitudes = np.fft.fft(sample_set)

    h_parameter = calculate_h(cps3, comb_x_values_j)
    final_amplitudes = fourier_amplitudes * h_parameter
    return np.abs(final_amplitudes)

def find_center(start_freq, first_harmon_width):
    """
    Find the center frequency for a comb.

    Parameters:
    start_freq: scalar
    first_harmon_width: scalar

    Returns:
    center: scalar
    """
    center = start_freq + (0.5 * first_harmon_width)
    return center

def trim_data(final_x_axis0, final_y_axis0, horizontal_comb_shift0, width_of_each_comb0):
    """
    Trim data to only include the relevant frequencies.

    Parameters:
    final_x_axis0: (n,)
    final_y_axis0: (n,)
    horizontal_comb_shift0: scalar
    width_of_each_comb0: scalar

    Returns:
    grand_update: list of trimmed x and y values
    """
    lower_bound_first_harmonic = horizontal_comb_shift0 - (0.5 * width_of_each_comb0)
    upper_bound_first_harmonic = horizontal_comb_shift0 + (0.5 * width_of_each_comb0)

    new_x_axis = []
    new_y_axis = []

    for individual in range(len(final_x_axis0)):
        if final_x_axis0[individual] >= lower_bound_first_harmonic and final_x_axis0[individual] < upper_bound_first_harmonic:
          new_x_axis.append(final_x_axis0[individual])
          new_y_axis.append(final_y_axis0[individual])

    grand_update = []
    grand_update.append(new_x_axis)
    grand_update.append(new_y_axis)
    return grand_update

# Trim Data to only include the relevant frequencies
startIndex = np.floor((startIndex - combFrequencies[0]) / freqGap).astype(int)

'''This is the for loop which generates the combs and creates the
data array in the desired format, which is 1000 rows with only the columns we want
from the combs. Each 100 rows is a distinct molecule in the ground truth. Data is the
array we attempt to cluster.'''
for i in range(IRUnprocessed.shape[0]-1):
    print('Current Row: ' + str(i + 1))
    # Generate all data
    progress = 0
    for guide in range(len(startIndex)):
        # Main parameters
        peak_spacing = teeth_spacing
        wavenumber_broadness = 3 * combLength
        horizontal_comb_shift = find_center(startIndex[guide], 20)
        noise_of_pulse = 0.00

        # Other parameters
        drift_comb = 0.000
        jitter_comb = 0.000
        refractive_index_comb = 000.0
        absorption_coefficient_comb = 0.0
        total_experiment_duration = 1e3

        broadness_of_comb = wavenumber_broadness / 100
        comb_parameters = {'rep_rate': peak_spacing,
                        'pulse_duration': 60e-3 * (1 / broadness_of_comb),
                        'time': total_experiment_duration,
                        'sample_rate': 100e0 * broadness_of_comb,
                        'noise': noise_of_pulse,
                        'jitter': jitter_comb,
                        'drift': drift_comb,
                        'n_0': refractive_index_comb,
                        'alpha_0': absorption_coefficient_comb}

        comb_x_axis = comb_x(comb_parameters)
        comb_y_axis = comb_y(comb_parameters, comb_x_axis)
        final_x_axis = comb_x_axis + horizontal_comb_shift
        final_y_axis = comb_y_axis / (np.max(comb_y_axis))

        new_values = trim_data(final_x_axis, final_y_axis, find_center(startIndex[guide], 20), 20)

        frequencies = IRUnprocessed[0]
        transmittance_values = IRUnprocessed[i+1] / (np.max(IRUnprocessed[i+1]))

        ir_spectrum_interpolated_values = np.interp(x = new_values[0], xp = frequencies, fp = transmittance_values)
        exiting_comb = ir_spectrum_interpolated_values * new_values[1]

        if guide == 0:
            column_names = []
            for ex in range(len(new_values[0])):
              name = f'comb_point_{ex}'
              column_names.append(name)
            export = pd.DataFrame(columns = column_names)

        sorted_indices = np.argsort(new_values[0])

        updated_x = []
        updated_y = []
        for finality in sorted_indices:
            updated_x.append(new_values[0][finality])
            updated_y.append(new_values[1][finality])

        export.loc[progress] = updated_x
        progress = progress + 1
        export.loc[progress] = updated_y
        progress = progress + 1
    dataRow = []
    for j in range(export.shape[0]):
        if j % 2 != 0:
            dataRow.extend(export.iloc[j].tolist())
    if i == 0:
        data = dataRow
    else:
        data = np.vstack((data, dataRow))
data = data[1:, :]

'''
This is the code to do portions of a wider comb instead of
the smaller combs, by just pruning the data. Old code, if you'd
like to use this please comment out everything in the comb section
above.

mask = np.zeros(data.shape[1], dtype=bool)
for start in startIndex:
    mask[start:start + originalCombLength] = True

data = data[:, mask]'''

# Determine the number of rows per molecule
rows_per_molecule = data.shape[0] // numMolecules

# Create labels
# labels: (number of samples,)
labels = np.hstack([[i] * rows_per_molecule for i in range(numMolecules)])

# Handle any remaining rows that might not have been labeled due to integer division
remaining_rows = data.shape[0] - len(labels)
if (remaining_rows) > 0:
    labels = np.hstack((labels, [numMolecules - 1] * remaining_rows))


'''This is where clustering starts. There are other methods I have tested
in another file and could easily add, but this has functionality for
PCA or T-SNE and then Spectral Clustering on the reduced dimenion, Spectral Clustering
on the full dataset, and HDBScan on the full dataset.'''
# Function to plot PCA or t-SNE results with a scatter plot, default is PCA
def plot_reduction_with_vectors(X, y, method='PCA', n_components=2, labels_legend=None):
    """
    Plot dimensionality reduction results with scatter plot.

    Parameters:
    X: (m, n)
    y: (m,)
    method: string ('PCA' or 't-SNE')
    n_components: scalar
    labels_legend: list of strings (optional)
    """
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Invalid dimensionality reduction method. Use 'PCA' or 't-SNE'.")

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    if n_components == 1:
        scatter = plt.scatter(X_reduced[:, 0], np.zeros_like(X_reduced[:, 0]), c=y, cmap='viridis')
        plt.xlabel(f'{method} Component 1')
    elif n_components == 2:
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis')
        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')
        ax.set_zlabel(f'{method} Component 3')

    plt.title(f'{method} with {n_components} Component{"s" if n_components > 1 else ""}')
    plt.colorbar(scatter)

    # Add legend
    if labels_legend:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, labels_legend, title="Datasets")

    plt.show()

# Plot PCA or t-SNE results for 1, 2, and 3 components
labels_legend = ['C2H2', 'CH4', 'NH3']
if dimensionality_reduction_method in [1, 2]:
    method = 'PCA' if dimensionality_reduction_method == 1 else 't-SNE'
    plot_reduction_with_vectors(data, labels, method=method, n_components=1, labels_legend=labels_legend)
    plot_reduction_with_vectors(data, labels, method=method, n_components=2, labels_legend=labels_legend)
    plot_reduction_with_vectors(data, labels, method=method, n_components=3, labels_legend=labels_legend)

'''
Misclustering rate code. I may look through this and make it more precise in
some edge cases, but generally it works in that a very high misclustering rate
implies the methods broke down and a very low one means it largely succeeded.
'''
# Function to calculate misclustered percentage
def calculate_misclustered_percentage(labels, clusters):
    """
    Calculate the percentage of misclustered points.

    Parameters:
    labels: (m,)
    clusters: (m,)

    Returns:
    misclustered_percentage: scalar
    """
    misclustered_counts = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = labels == label
        cluster_labels, counts = np.unique(clusters[label_indices], return_counts=True)
        majority_count = np.max(counts)
        misclustered_counts.append(np.sum(counts) - majority_count)
    total_misclustered = np.sum(misclustered_counts)
    total_points = len(labels)
    misclustered_percentage = (total_misclustered / total_points) * 100
    return misclustered_percentage

# Function to calculate misclustered percentage and outlier percentage
def calculate_misclustered_and_outlier_percentage(labels, clusters):
    """
    Calculate the percentage of misclustered points and outliers.

    Parameters:
    labels: (m,)
    clusters: (m,)

    Returns:
    misclustered_percentage: scalar
    outlier_percentage: scalar
    """
    cluster_counts = Counter(clusters)
    if -1 in cluster_counts:
        del cluster_counts[-1]  # Ignore the noise cluster if it exists
    largest_clusters = cluster_counts.most_common(10)
    largest_cluster_indices = {cluster[0] for cluster in largest_clusters}

    non_outlier_indices = np.isin(clusters, list(largest_cluster_indices))
    non_outlier_labels = labels[non_outlier_indices]
    non_outlier_clusters = clusters[non_outlier_indices]
    outlier_percentage = (1 - np.sum(non_outlier_indices) / len(labels)) * 100
    misclustered_percentage = calculate_misclustered_percentage(non_outlier_labels, non_outlier_clusters)

    return misclustered_percentage, outlier_percentage

# Runs spectral clustering multiple times and compute the average misclustered percentage
def run_spectral_clustering_and_evaluate(X, labels, n_clusters, n_runs=5):
    """
    Run spectral clustering multiple times and compute the average misclustered percentage.

    Parameters:
    X: (m, n)
    labels: (m,)
    n_clusters: scalar
    n_runs: scalar

    Returns:
    mean_misclustered_percentage: scalar
    """
    misclustered_percentages = []
    for _ in range(n_runs):
        sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
        clusters = sc.fit_predict(X)
        misclustered_percentage = calculate_misclustered_percentage(labels, clusters)
        misclustered_percentages.append(misclustered_percentage)
    misclustered_percentages = sorted(misclustered_percentages)[1:-1]
    return np.mean(misclustered_percentages)

# Runs HDBSCAN and computes the misclustered percentage
def run_hdbscan_and_evaluate(X, labels, min_cluster_size=5):
    """
    Run HDBSCAN and compute the misclustered percentage.

    Parameters:
    X: (m, n)
    labels: (m,)
    min_cluster_size: scalar

    Returns:
    misclustered_percentage: scalar
    outlier_percentage: scalar
    """
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = hdbscan.fit_predict(X)
    return calculate_misclustered_and_outlier_percentage(labels, clusters)

# Applies dimensionality reduction and evaluate for 1D, 2D, and 3D
if dimensionality_reduction_method == 3:
    mean_misclustered_percentage = run_spectral_clustering_and_evaluate(data, labels, n_clusters=numMolecules)
    print(f"Percentage of misclustered points (Spectral Clustering): {mean_misclustered_percentage:.2f}%")
elif dimensionality_reduction_method == 4:
    misclustered_percentage, outlier_percentage = run_hdbscan_and_evaluate(data, labels)
    print(f"Percentage of misclustered points (HDBSCAN): {misclustered_percentage:.2f}%, Outliers: {outlier_percentage:.2f}%")
else:
    dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for dim in dimensions:
        if dimensionality_reduction_method == 1:
            reducer = PCA(n_components=dim)
        elif dimensionality_reduction_method == 2 and dim < 4:
            reducer = TSNE(n_components=dim)
        else:
            continue

        X_reduced = reducer.fit_transform(data)
        mean_misclustered_percentage = run_spectral_clustering_and_evaluate(X_reduced, labels, n_clusters=numMolecules)
        print(f"Percentage of misclustered points ({dim}D {method}): {mean_misclustered_percentage:.2f}%")
