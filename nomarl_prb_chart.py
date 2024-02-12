import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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

    # Interpretation based on the p-value
    if p_value > 0.05:
        print("Data appears to follow a normal distribution.")
    else:
        print("Data does not appear to follow a normal distribution.")


filename = "Data.xlsx"  # Replace with your actual file path
sheet_name = "Sheet1"  # Replace with the correct sheet name if needed
col_index = 4  # Adjust the CL index based on your data structure
normal_probability_plot(filename, sheet_name, col_index)
