import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.stats as stats

file_path = 'Data.xlsx'
sheet_name = 'Sheet1'

def plot_hist(file_path=file_path, sheet_name=sheet_name, col_index=None):
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

    # Plot histogram
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(data, bins=9, color='#7DA7D9', edgecolor='black', alpha=0.7)

    max_freq = max(n)
    scaling_factor = max_freq / stats.norm.pdf(mean, mean, overall_std)

    lsl = mean - 3*overall_std
    usl = mean + 3*overall_std

    # Plot overall normal curve
    x = np.linspace(lsl, usl, 200)
    p_overall = stats.norm.pdf(x, mean, overall_std) * scaling_factor
    plt.plot(x, p_overall, 'k', linewidth=2, label='Overall STD', color='r')

    # Add labels and title
    plt.xlabel('Hist Avg')
    plt.ylabel('Frequency')
    plt.title('Capability Histogram')

    # Add legend
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()

col_index=4
plot_hist(col_index=col_index)