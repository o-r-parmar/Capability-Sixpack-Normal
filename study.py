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
    p = stats.norm.pdf(x, mean, 10) * scaling_factor
    plt.plot(x, p, 'r', linewidth=2)

    # Add labels and title
    plt.xlabel(f'Column Index {col_index}')
    plt.ylabel('Frequency')
    plt.title('Capability Histogram')
    plt.show()

plot_hist(col_index=1)