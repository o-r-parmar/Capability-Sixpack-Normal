import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

file_path = 'Data.xlsx'
sheet_name = 'Sheet1'

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
    n, bins, patches = ax1.hist(data, bins=9, color='#7DA7D9', edgecolor='black', alpha=0.7)
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

    # Create table in the second subplot
    cell_text = [[f"{mean:.2f}"], [f"{overall_std:.2f}"], [f"{n_value}"]]
    row_labels = ['Mean', 'Std Dev', 'N']
    ax2.axis('off')  # Hide axis for the table
    table = ax2.table(cellText=cell_text,
                      rowLabels=row_labels,
                      bbox=[0.25,0.7,0.5,0.3])

    # Adjust layout to ensure no overlap
        
    plt.tight_layout()

    # Display the plot and table
    plt.show()
    
plot_hist(file_path, sheet_name, col_index=3)
plot_hist(file_path, sheet_name, col_index=4)
plot_hist(file_path, sheet_name, col_index=5)
