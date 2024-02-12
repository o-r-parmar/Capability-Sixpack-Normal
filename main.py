import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_hist(file_path, sheet_name, col_index=None):
    # Read the specified Excel file and sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Verify column index is provided
    if col_index is None:
        raise ValueError("Column index must be provided")
    
    # Access column by index and drop NA values
    data = df.iloc[:, col_index].dropna().values

    # Calculate statistics
    mean = np.mean(data)
    overall_std = np.std(data, ddof=1)
    n_value = len(data)  # Number of observations

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot histogram in the first subplot
    n, bins, patches = ax1.hist(data, bins=9, color='#7DA7D9', edgecolor='#0000FF', alpha=0.7)
    ax1.set_xlabel('Hist Avg')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Capability Histogram')

    # Calculate and plot LSL, USL, and normal curve on ax1
    lsl = mean - 3*overall_std
    usl = mean + 3*overall_std
    margin = 0.2 * (usl - lsl)  # Margin to ensure some space before and after LSL and USL lines
    #plot LSL USL
    ax1.axvline(x=lsl, color='r', linestyle='--', linewidth=2, label='LSL')
    ax1.axvline(x=usl, color='r', linestyle='--', linewidth=2, label='USL')
    
    # Adjust ax1 view limits
    ax1.set_xlim([lsl - margin, usl + margin])

    #plot normalized curve
    x = np.linspace(lsl, usl, 200)
    scaling_factor = max(n) / stats.norm.pdf(mean, mean, overall_std)
    p_overall = stats.norm.pdf(x, mean, overall_std) * scaling_factor
    ax1.plot(x, p_overall, 'r', linewidth=2, label='Overall STD')
    ax1.legend(loc='upper right')

    # # Create table in the second subplot
    # cell_text = [[f"{mean:.2f}"], [f"{overall_std:.2f}"], [f"{n_value}"]]
    # row_labels = ['Mean', 'Std Dev', 'N']
    # ax2.axis('off')  # Hide axis for the table
    # table = ax2.table(cellText=cell_text,
    #                   rowLabels=row_labels,
    #                   bbox=[0.25,0.7,0.5,0.3])

    # Adjust layout to ensure no overlap
        
    plt.tight_layout()

    # Display the plot and table
    plt.show()

def i_chart(filename, sheet_name, col_index, sub_group=1):
    """
    Generates an I chart with out-of-control points highlighted in red. Adds a margin for LSL and USL 
    that is always 30% above and below the maximum and minimum values on the Y-axis.

    Args:
        filename (str): The path to the XLSX file.
        sheet_name (str, optional): The name of the sheet containing the data. Defaults to None.
        cl_index (int): The index of the CL (column) containing the observations.

    Returns:
        None.
    """
    if(sub_group!=1):
        raise ValueError("Subgroup Size has to be 1 for IChart Analysis")

    # Read data from the XLSX file
    df = pd.read_excel(filename, sheet_name=sheet_name)

    # Select data based on CL index
    data = df.iloc[:, col_index].tolist()

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Calculate control limits
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    plt.figure(figsize=(12, 6))

    # Plot individual data points
    for i, value in enumerate(data):
        color = 'red' if (value < lcl or value > ucl) else 'blue'
        plt.plot(i, value, 'o', markersize=5, color=color)

    # Plot lines
    plt.plot(range(len(data)), data, linestyle='-', color='blue')  # Connect data points

    # Plot center line and control limits
    plt.axhline(y=mean, color='g', linestyle='-', label='Mean Line')
    plt.axhline(y=ucl, color='r', linestyle='--', label='Upper Control Limit')
    plt.axhline(y=lcl, color='r', linestyle='--', label='Lower Control Limit')

    # Determine plot limits with margins
    y_min, y_max = plt.ylim()  # Get current Y-axis limits
    y_range = y_max - y_min
    margin = 0.3 * y_range  # 30% margin

    plt.ylim([y_min - margin, y_max + margin])  # Set Y-axis limits with margin on both sides

    # Set labels and title
    plt.xlabel('#Observation')
    plt.ylabel(df.columns[col_index])
    plt.title('I Chart with Out-of-Control Points Highlighted and Margin for LSL and USL')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def mr_chart(filename, sheet_name, col_index, sub_group=1):
    """
    Generates an MR chart with out-of-control points highlighted. This function is designed for
    individual measurements (subgroup size = 1) and plots the moving range of the process.

    Args:
        filename (str): The path to the XLSX file.
        sheet_name (str, optional): The name of the sheet containing the data. Defaults to None.
        col_index (int): The index of the column containing the observations.

    Returns:
        None.
    """
    if(sub_group!=1):
        raise ValueError("Subgroup Size has to be 1 for IChart Analysis")

    # Read data from the XLSX file
    df = pd.read_excel(filename, sheet_name=sheet_name)

    # Select data based on column index
    data = df.iloc[:, col_index].tolist()

    # Calculate moving ranges
    moving_ranges = [abs(data[i] - data[i-1]) for i in range(1, len(data))]

    # Calculate the average moving range
    MR_mean = np.mean(moving_ranges)

    # Estimate sigma using the average moving range (considering subgroup size = 1)
    sigma_estimated = MR_mean / 1.128  # d2 factor for n=2

    # Calculate control limits
    upper_cl = MR_mean + 3 * sigma_estimated
    lower_cl = MR_mean - 3 * sigma_estimated if MR_mean - 3 * sigma_estimated > 0 else 0

    plt.figure(figsize=(12, 6))

    # Plot moving range data points
    for i, value in enumerate(moving_ranges):
        color = 'red' if (value < lower_cl or value > upper_cl) else 'blue'
        plt.plot(i, value, 'o', markersize=5, color=color)

    # Connect the dots with solid lines
    plt.plot(range(len(moving_ranges)), moving_ranges, linestyle='-', color='blue')

    # Plot center line and control limits
    plt.axhline(y=MR_mean, color='g', linestyle='-', label='Center Line')
    plt.axhline(y=upper_cl, color='r', linestyle='--', label='Upper Control Limit')
    if lower_cl > 0:  # Plot lower control limit if it is above zero
        plt.axhline(y=lower_cl, color='r', linestyle='--', label='Lower Control Limit')

    # Set labels and title
    plt.xlabel('#Observation')
    plt.ylabel('Moving Range')
    plt.title('MR Chart with Out-of-Control Points Highlighted')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def normal_probability_plot(file_name, sheet_name, col_index):
    """
    Generates a normal probability plot, including approximate confidence bounds, and performs
    the Anderson-Darling test for normality.

    Args:
        file_name (str): The path to the Excel file.
        sheet_name (str): The name of the sheet within the Excel file.
        col_index (int): The index of the column containing the data to be analyzed.

    Returns:
        None.
    """
    # Read data from Excel
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    data = df.iloc[:, col_index]
    data_sorted = np.sort(data)

    # Calculate theoretical quantiles and actual data quantiles
    theoretical_quantiles = stats.norm.ppf((np.arange(1, len(data) + 1) - 0.5) / len(data))
    actual_quantiles = (data_sorted - np.mean(data_sorted)) / np.std(data_sorted, ddof=1)

    # Perform Anderson-Darling test for normality
    ad_stat, critical_values, p_value = stats.anderson(data_sorted, dist='norm')

    # Plot the actual data quantiles vs. theoretical quantiles
    plt.figure(figsize=(8, 6))
    plt.plot(theoretical_quantiles, actual_quantiles, 'o', label='Data Points')

    # Plot the expected line (45-degree line)
    plt.plot(theoretical_quantiles, theoretical_quantiles, 'r-', label='Expected Normal')

    # Approximate confidence bounds (for illustration)
    se = 1.96 * np.sqrt(len(data))  # Simplified standard error approximation, adjust as necessary
    upper_bound = theoretical_quantiles + se
    lower_bound = theoretical_quantiles - se
    plt.plot(theoretical_quantiles, upper_bound, 'g--', label="Upper Confidence Bound (approx.)")
    plt.plot(theoretical_quantiles, lower_bound, 'b--', label="Lower Confidence Bound (approx.)")

    # Add labels and title
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Ordered Values')
    plt.title('Normal Probability Plot with Approximate Confidence Bounds')
    plt.legend()

    # Display the Anderson-Darling test statistic and p-value
    # plt.figtext(0.5, -0.05, f'AD Test Statistic: {ad_stat:.2f}, p-value: {p_value:.3f}', 
    #             ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Adjust layout to make room for the figtext
    plt.subplots_adjust(bottom=0.15)

    # Show the plot
    plt.show()

filename = "Data.xlsx"
sheet_name = "Sheet1"
col_index = [3,4,5]

for i in col_index:
    plot_hist(filename, sheet_name, col_index=i)
    print(i)
    i_chart(filename, sheet_name, col_index=i)
    print(i)
    mr_chart(filename, sheet_name, col_index=i)
    print(i)
    normal_probability_plot(filename, sheet_name, col_index=i)
    print(i)


