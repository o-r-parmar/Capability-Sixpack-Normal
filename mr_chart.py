import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

filename = "Data.xlsx"  # Replace with your actual file path
sheet_name = "Sheet1"  # Replace with the correct sheet name if needed
col_index = 4  # Adjust the CL index based on your data structure
mr_chart(filename, sheet_name, col_index)
