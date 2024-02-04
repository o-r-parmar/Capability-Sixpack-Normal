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

col_index=4
plot_xbar(col_index=col_index)
plot_i_chart(col_index=col_index)
plot_hist(col_index=col_index)