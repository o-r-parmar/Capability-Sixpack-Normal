import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import scipy.stats as stats

file_path = 'Data.xlsx'
sheet_name = 'Sheet1'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    upper_cl = mean + 3 * std
    lower_cl = mean - 3 * std

    plt.figure(figsize=(12, 6))

    # Plot individual data points
    for i, value in enumerate(data):
        color = 'red' if (value < lower_cl or value > upper_cl) else 'blue'
        plt.plot(i, value, 'o', markersize=5, color=color)

    # Plot lines
    plt.plot(range(len(data)), data, linestyle='-', color='blue')  # Connect data points

    # Plot center line and control limits
    plt.axhline(y=mean, color='g', linestyle='-', label='Mean Line')
    plt.axhline(y=upper_cl, color='r', linestyle='--', label='Upper Control Limit')
    plt.axhline(y=lower_cl, color='r', linestyle='--', label='Lower Control Limit')

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

# Example usage:
filename = "Data.xlsx"  # Replace with your actual file path
sheet_name = "Sheet1"  # Replace with the correct sheet name if needed
cl_index = 4  # Adjust the CL index based on your data structure
i_chart(filename, sheet_name, cl_index)