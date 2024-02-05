import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.stats as stats

file_path = 'Data.xlsx'
sheet_name = 'Sheet1'

def plot_hist(file_path=file_path, sheet_name=sheet_name, col_index=None, subgroup_size=1):
    # Read the specified Excel file and sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    #Verify all inputs are ok
    if col_index is not None:
        data = df.iloc[:, col_index].dropna().values  # Access column by index and drop NA values
    else:
        raise ValueError("Column index must be provided")

    # Calculate statistics
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Plot histogram
    plt.figure(figsize=(15, 6))
    n, bins, patches = plt.hist(data, bins=30, color='#7DA7D9', edgecolor='black', alpha=0.7, density=False)

    # Calculate the scaling factor (total count * bin width)
    bin_width = bins[1] - bins[0]  # Width of one bin
    scaling_factor = len(data) * bin_width

    # Plot overall normal curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mean, std) * scaling_factor
    plt.plot(x, p, 'r', linewidth=2)

    # Add labels and title
    plt.xlabel(f'Column Index {col_index}')
    plt.ylabel('Frequency')
    plt.title('Capability Histogram')
    plt.show()

def plot_xbar(file_path=file_path, sheet_name=sheet_name, col_index=None):
    if col_index is None:
        raise ValueError("Column index must be provided")
    
    # Read the specified Excel file, sheet, and column
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    subgroup_means = df.iloc[:, col_index].dropna().values
    
    # from the means of these subgroups.
    std_dev = np.std(subgroup_means, ddof=1)
    CL = np.mean(subgroup_means)
    UCL = CL + 3 * std_dev
    LCL = CL - 3 * std_dev
    
    # Set up the control chart
    subgroup_order = np.arange(len(subgroup_means)) + 1
    plt.figure(figsize=(12, 6))
    plt.plot(subgroup_order, subgroup_means, marker='o', linestyle='-', color='#7DA7D9')
    plt.axhline(CL, color='green', linestyle='-', label=f'CL={CL:.3f}')
    plt.axhline(UCL, color='red', linestyle='--', label=f'UCL={UCL:.3f}')
    plt.axhline(LCL, color='red', linestyle='--', label=f'LCL={LCL:.3f}')

    # Highlight out-of-control points
    for i, mean in enumerate(subgroup_means):
        if mean > UCL or mean < LCL:
            plt.plot(subgroup_order[i], mean, marker='o', color='red')

    # Add labels and title
    plt.xlabel('Subgroup')
    plt.ylabel('Sample Mean')
    plt.title('X-bar Control Chart')
    plt.legend()
    plt.show()

def plot_i_chart(file_path=file_path, sheet_name=sheet_name, col_index=None):
    if col_index is None:
        raise ValueError("Column index must be provided")

    # Read the specified Excel file, sheet, and column
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    individual_values = df.iloc[:, col_index].dropna().values

    # Calculate control limits and center line
    CL = np.mean(individual_values)
    std_dev = np.std(individual_values, ddof=1)
    UCL = CL + 3 * std_dev
    LCL = CL - 3 * std_dev
    
    # Set up the control chart
    measurement_order = np.arange(len(individual_values)) + 1
    plt.figure(figsize=(12, 6))
    plt.plot(measurement_order, individual_values, marker='o', linestyle=':', color='#7DA7D9')
    plt.axhline(CL, color='green', linestyle='-', label=f'CL={CL:.3f}')
    plt.axhline(UCL, color='red', linestyle='--', label=f'UCL={UCL:.3f}')
    plt.axhline(LCL, color='red', linestyle='--', label=f'LCL={LCL:.3f}')

    # Highlight out-of-control points
    for i, value in enumerate(individual_values):
        if value > UCL or value < LCL:
            plt.plot(measurement_order[i], value, marker='o', color='red')

    # Add labels and title
    plt.xlabel('Measurement Number')
    plt.ylabel('Individual Value')
    plt.title('I Chart')
    plt.legend()

    # Show the control chart
    plt.show()

#Some updates might be required but not at the moment becuase most of our inspections have only 1 subgroup
def plot_r_chart(file_path=file_path, sheet_name=sheet_name, col_index=None, subgroup_size=None):
    if col_index is None or subgroup_size is None:
        raise ValueError("Column index must be provided")
    if subgroup_size < 2:
        raise ValueError("Subgroup size must be greater than 1 for range calculation")

    # Read the specified Excel file, sheet, and column
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data = df.iloc[:, col_index].dropna().values

    # Calculate the ranges of subgroups
    subgroup_ranges = [np.ptp(data[i:i+subgroup_size]) for i in range(0, len(data), subgroup_size) if i+subgroup_size <= len(data)]

    # Calculate the center line (average range)
    mean = np.mean(subgroup_ranges)

    # Estimate standard deviation assuming subgroup size is large enough for the approximation to be reasonable
    d2 = 1.128  # d2 factor for subgroup size of 2; this varies depending on subgroup size
    std_dev = mean / d2

    # Calculate control limits
    UCL = mean + 3 * std_dev
    LCL = max(mean - 3 * std_dev, 0)  # Range can't be negative

    # Set up the control chart
    subgroup_order = np.arange(1, len(subgroup_ranges) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(subgroup_order, subgroup_ranges, marker='o', linestyle='-', color='#7DA7D9')
    plt.axhline(mean, color='green', linestyle='-', label=f'R̄={mean:.3f}')
    plt.axhline(UCL, color='red', linestyle='--', label=f'UCL={UCL:.3f}')
    plt.axhline(LCL, color='red', linestyle='--', label=f'LCL={LCL:.3f}')

    # Highlight out-of-control points
    for i, rng in enumerate(subgroup_ranges):
        if rng > UCL or rng < LCL:
            plt.plot(subgroup_order[i], rng, marker='o', color='red')

    # Add labels and title
    plt.xlabel('Subgroup')
    plt.ylabel('Range')
    plt.title('R Chart')
    plt.legend()

    # Show the control chart
    plt.show()

def plot_s_chart(file_path=file_path, sheet_name=sheet_name, col_index=None, subgroup_size=None):
    if col_index is None or subgroup_size is None:
        raise ValueError("Column index must be provided")

    # Read the specified Excel file, sheet, and column
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data = df.iloc[:, col_index].dropna().values

    # Calculate the standard deviations of subgroups
    subgroup_stds = [np.std(data[i:i+subgroup_size], ddof=1) for i in range(0, len(data), subgroup_size) if i+subgroup_size <= len(data)]

    # Calculate the center line (average of subgroup standard deviations)
    Sbar = np.mean(subgroup_stds)
    # Control limits for S Chart, which are typically set at B3 and B4 constants times Sbar, depending on subgroup size
    # For subgroup_size >= 9, we can use approximation B3 ≈ 1 and B4 ≈ 1, else consult specific tables for exact values.
    UCL = Sbar + 3 * Sbar / np.sqrt(subgroup_size)
    LCL = max(Sbar - 3 * Sbar / np.sqrt(subgroup_size), 0)  # Standard deviation can't be negative

    # Set up the control chart
    subgroup_order = np.arange(1, len(subgroup_stds) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(subgroup_order, subgroup_stds, marker='o', linestyle='-', color='#7DA7D9')
    plt.axhline(Sbar, color='green', linestyle='-', label=f'S̄={Sbar:.5f}')
    plt.axhline(UCL, color='red', linestyle='--', label=f'UCL={UCL:.5f}')
    plt.axhline(LCL, color='red', linestyle='--', label=f'LCL={LCL:.5f}')

    # Highlight out-of-control points
    for i, std in enumerate(subgroup_stds):
        if std > UCL or std < LCL:
            plt.plot(subgroup_order[i], std, marker='o', color='red')

    # Add labels and title
    plt.xlabel('Subgroup')
    plt.ylabel('Standard Deviation')
    plt.title('S Chart')
    plt.legend()

    # Show the control chart
    plt.show()

#Updates to be made
def interpret_s_chart(subgroup_stds, Sbar, UCL, LCL):
    """
    Interpret an S chart for special causes.

    Parameters:
    subgroup_stds (list): List of standard deviations for each subgroup.
    Sbar (float): Centerline value for standard deviations.
    UCL (float): Upper Control Limit for standard deviations.
    LCL (float): Lower Control Limit for standard deviations.

    Returns:
    list: A list of strings indicating the results of the special cause tests.
    """

    results = []

    # Test 1: One point more than 3 standard deviations from center line
    for i, std in enumerate(subgroup_stds):
        if std > UCL or std < LCL:
            results.append(f"Test 1: Subgroup {i+1} has a point more than 3 standard deviations from the center line")

    # Test 2: Nine points in a row on the same side of the center line
    consecutive_count = 0
    for std in subgroup_stds:
        if std > Sbar or std < Sbar:
            consecutive_count += 1
            if consecutive_count >= 9:
                results.append("Test 2: Nine points in a row on the same side of the center line")
        else:
            consecutive_count = 0

    # Test 3: Six points in a row, all increasing or all decreasing
    increasing_count = 0
    decreasing_count = 0
    for i in range(len(subgroup_stds)):
        if i > 0:
            if subgroup_stds[i] > subgroup_stds[i-1]:
                increasing_count += 1
                decreasing_count = 0
            elif subgroup_stds[i] < subgroup_stds[i-1]:
                decreasing_count += 1
                increasing_count = 0
            else:
                increasing_count = 0
                decreasing_count = 0

            if increasing_count >= 6:
                results.append("Test 3: Six points in a row, all increasing")
            elif decreasing_count >= 6:
                results.append("Test 3: Six points in a row, all decreasing")

    # Test 4: Fourteen points in a row, alternating up and down
    alternating_count = 0
    for i in range(len(subgroup_stds)):
        if i > 0:
            if (subgroup_stds[i] > subgroup_stds[i-1] and i % 2 == 0) or \
               (subgroup_stds[i] < subgroup_stds[i-1] and i % 2 != 0):
                alternating_count += 1
            else:
                alternating_count = 0

            if alternating_count >= 14:
                results.append("Test 4: Fourteen points in a row, alternating up and down")

    # Test 5: Two out of three points more than 2 standard deviations from center line (same side)
    consecutive_count = 0
    for i, std in enumerate(subgroup_stds):
        if std > UCL or std < LCL:
            consecutive_count += 1
            if consecutive_count >= 2 and i < len(subgroup_stds) - 1:
                if subgroup_stds[i+1] > UCL or subgroup_stds[i+1] < LCL:
                    results.append(f"Test 5: Subgroup {i+1} and Subgroup {i+2} have two out of three points more than 2 standard deviations from the center line")
            elif i < len(subgroup_stds) - 1:
                if (subgroup_stds[i+1] > UCL or subgroup_stds[i+1] < LCL) and \
                   (subgroup_stds[i+2] > UCL or subgroup_stds[i+2] < LCL):
                    results.append(f"Test 5: Subgroup {i+2} and Subgroup {i+3} have two out of three points more than 2 standard deviations from the center line")
        else:
            consecutive_count = 0

    # Test 6: Four out of five points more than 1 standard deviation from center line (same side)
    consecutive_count = 0
    for i, std in enumerate(subgroup_stds):
        if std > UCL or std < LCL:
            consecutive_count += 1
            if consecutive_count >= 4 and i < len(subgroup_stds) - 1:
                if subgroup_stds[i+1] > UCL or subgroup_stds[i+1] < LCL:
                    results.append(f"Test 6: Subgroup {i+1} to Subgroup {i+4} have four out of five points more than 1 standard deviation from the center line")
        else:
            consecutive_count = 0

    # Test 7: Fifteen points in a row within 1 standard deviation of center line (either side)
    consecutive_count = 0
    for i, std in enumerate(subgroup_stds):
        if abs(std - Sbar) <= Sbar:
            consecutive_count += 1
            if consecutive_count >= 15:
                results.append("Test 7: Fifteen points in a row within 1 standard deviation of the center line (either side)")
        else:
            consecutive_count = 0

    # Test 8: Eight points in a row more than 1 standard deviation from center line (either side)
    consecutive_count = 0
    for i, std in enumerate(subgroup_stds):
        if std > UCL or std < LCL:
            consecutive_count += 1
            if consecutive_count >= 8:
                results.append("Test 8: Eight points in a row more than 1 standard deviation from the center line (either side)")
        else:
            consecutive_count = 0

    return results

col_index=1
subgroup_size = 9

plot_hist(col_index=col_index)
plot_i_chart(col_index=col_index)