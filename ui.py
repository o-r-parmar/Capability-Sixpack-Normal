import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys

def plot_hist(file_path, sheet_name, col_index=None):
    '''
    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Worksheet name.
        col_index (int): Column index for plotting.
    Raises:
        ValueError: If col_index is not provided.
    Creates:
        Image file: histogram_plot_{col_index}.png in the "images" directory.
    Returns:
        None.
    '''
    if col_index is None:
        raise ValueError("Column index must be provided")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    data = df.iloc[:, col_index].dropna().values

    mean = np.mean(data)
    overall_std = np.std(data, ddof=1)
    
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(data, bins=9, color='#7DA7D9', edgecolor='black', alpha=0.7)
    plt.xlabel('Histogram Average')
    plt.ylabel(df.columns[col_index])
    plt.title('Capability Histogram')

    lsl = mean - 3*overall_std
    usl = mean + 3*overall_std
    margin = 0.2 * (usl - lsl)

    plt.axvline(x=lsl, color='r', linestyle='--', linewidth=2, label='LSL')
    plt.axvline(x=usl, color='r', linestyle='--', linewidth=2, label='USL')
    
    plt.xlim([lsl - margin, usl + margin])

    x = np.linspace(lsl, usl, 200)
    scaling_factor = max(n) / stats.norm.pdf(mean, mean, overall_std)
    p_overall = stats.norm.pdf(x, mean, overall_std) * scaling_factor
    plt.plot(x, p_overall, 'r', linewidth=2, label='Overall STD')
    plt.legend(loc='upper right')

    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    plt.savefig(f"{images_dir}/histogram_plot_{col_index}.png", dpi=600)

def i_chart(filename, sheet_name, col_index, sub_group=1):
    '''
    Args:
        filename (str): Path to the XLSX file.
        sheet_name (str): Worksheet name.
        col_index (int): Column index for data.
        sub_group (int, optional): Subgroup size, default is 1.
    Raises:
        ValueError: If sub_group is not 1.
    Creates:
        Image file: i_chart_{col_index}.png in the "images" directory.
    Returns:
        None.
    '''
    if(sub_group!=1):
        raise ValueError("Subgroup Size has to be 1 for IChart Analysis")

    df = pd.read_excel(filename, sheet_name=sheet_name)

    data = df.iloc[:, col_index].tolist()

    mean = np.mean(data)
    std = np.std(data)
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    plt.figure(figsize=(12, 6))

    for i, value in enumerate(data):
        color = 'red' if (value < lcl or value > ucl) else 'blue'
        plt.plot(i, value, 'o', markersize=5, color=color)

    plt.plot(range(len(data)), data, linestyle='-', color='blue')
    plt.axhline(y=mean, color='g', linestyle='-', label='Mean Line')
    plt.axhline(y=ucl, color='r', linestyle='--', label='Upper Control Limit')
    plt.axhline(y=lcl, color='r', linestyle='--', label='Lower Control Limit')

    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    margin = 0.3 * y_range

    plt.ylim([y_min - margin, y_max + margin])

    plt.xlabel('#Observation')
    plt.ylabel(df.columns[col_index])
    plt.title('I Chart with Out-of-Control Points Highlighted and Margin for LSL and USL')

    plt.legend()

    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    plt.savefig(f"{images_dir}/i_chart_{col_index}.png", dpi=600)

def mr_chart(filename, sheet_name, col_index, sub_group=1):
    '''
    Args:
        filename (str): Path to the XLSX file.
        sheet_name (str): Worksheet name.
        col_index (int): Column index for data.
        sub_group (int, optional): Subgroup size, default is 1.
    Raises:
        ValueError: If sub_group is not 1.
    Creates:
        Image file: mr_chart_{col_index}.png in the "images" directory.
    Returns:
        None.
    '''
    if(sub_group!=1):
        raise ValueError("Subgroup Size has to be 1 for IChart Analysis")

    df = pd.read_excel(filename, sheet_name=sheet_name)

    data = df.iloc[:, col_index].tolist()

    moving_ranges = [abs(data[i] - data[i-1]) for i in range(1, len(data))]

    mr_mean = np.mean(moving_ranges)

    sigma_estimated = mr_mean / 1.128

    ucl = mr_mean + 3 * sigma_estimated
    lcl = mr_mean - 3 * sigma_estimated if mr_mean - 3 * sigma_estimated > 0 else 0

    plt.figure(figsize=(12, 6))

    for i, value in enumerate(moving_ranges):
        color = 'red' if (value < lcl or value > ucl) else 'blue'
        plt.plot(i, value, 'o', markersize=5, color=color)

    plt.plot(range(len(moving_ranges)), moving_ranges, linestyle='-', color='blue')

    plt.axhline(y=mr_mean, color='g', linestyle='-', label='Center Line')
    plt.axhline(y=ucl, color='r', linestyle='--', label='Upper Control Limit')
    if lcl > 0:
        plt.axhline(y=lcl, color='r', linestyle='--', label='Lower Control Limit')

    plt.xlabel('#Observation')
    plt.ylabel(df.columns[col_index])
    plt.title('MR Chart with Out-of-Control Points Highlighted')
    plt.legend()

    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    plt.savefig(f"{images_dir}/mr_chart_{col_index}.png", dpi=600)

def normal_probability_plot(file_name, sheet_name, col_index):
    '''
    Args:
        file_name (str): Path to the Excel file.
        sheet_name (str): Worksheet name.
        col_index (int): Column index for analysis.
    Creates:
        Image file: normal_probability_plot_{col_index}.png in the "images" directory.
    Returns:
        None.
    '''
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    data = df.iloc[:, col_index]
    data_sorted = np.sort(data)

    theoretical_quantiles = stats.norm.ppf((np.arange(1, len(data) + 1) - 0.5) / len(data))
    actual_quantiles = (data_sorted - np.mean(data_sorted)) / np.std(data_sorted, ddof=1)

    ad_stat, critical_values, p_value = stats.anderson(data_sorted, dist='norm')

    plt.figure(figsize=(12, 6))
    plt.plot(theoretical_quantiles, actual_quantiles, 'o', label='Data Points')

    plt.plot(theoretical_quantiles, theoretical_quantiles, 'r-', label='Expected Normal')

    se = 1.96 * np.sqrt(len(data))
    upper_bound = theoretical_quantiles + se
    lower_bound = theoretical_quantiles - se
    plt.plot(theoretical_quantiles, upper_bound, 'g--', label="Upper Confidence Bound (approx.)")
    plt.plot(theoretical_quantiles, lower_bound, 'b--', label="Lower Confidence Bound (approx.)")

    plt.xlabel('Theoretical Quantiles')
    plt.ylabel(df.columns[col_index])
    plt.title('Normal Probability Plot with Approximate Confidence Bounds')
    plt.legend()
    plt.subplots_adjust(bottom=0.15)

    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    plt.savefig(f"{images_dir}/normal_probability_plot_{col_index}.png", dpi=600)

class ExcelColumnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel Column Selector and Plotter")
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.path_label = tk.Label(self.frame, text="Excel Path:")
        self.path_label.grid(row=0, column=0, sticky='w')
        self.path_entry = tk.Entry(self.frame, width=50)
        self.path_entry.grid(row=0, column=1, sticky='we')
        self.browse_button = tk.Button(self.frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2)

        self.load_button = tk.Button(self.frame, text="Load Columns", command=self.load_columns)
        self.load_button.grid(row=1, column=0, columnspan=3, pady=(5,0))

        self.columns_frame = tk.LabelFrame(self.root, text="Select Columns")
        self.columns_frame.pack(fill="both", expand="yes", padx=10, pady=5)

        self.done_button = tk.Button(self.root, text="Done", command=self.generate_plots)
        self.done_button.pack(pady=(0,10))

        self.column_vars = {}

    def exit_app(self):
        for var in self.column_vars.values():
            var.set(False)
        sys.exit()
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, file_path)

    def load_columns(self):
        file_path = self.path_entry.get()
        try:
            df = pd.read_excel(file_path)
            for widget in self.columns_frame.winfo_children():
                widget.destroy()
            self.column_vars = {}
            for i, col in enumerate(df.columns):
                self.column_vars[col] = tk.BooleanVar()
                cb = tk.Checkbutton(self.columns_frame, text=col, variable=self.column_vars[col])
                cb.pack(anchor="w")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Excel file: {e}")

    def generate_plots(self):
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        file_path = self.path_entry.get()
        sheet_name = 'Sheet1'  # Assuming default sheet name or prompt user to enter/select
        if not selected_columns:
            messagebox.showinfo("Info", "No columns selected.")
        else:
            for col_name in selected_columns:
                df = pd.read_excel(file_path)
                col_index = df.columns.get_loc(col_name)
                plot_hist(file_path, sheet_name, col_index)
                i_chart(file_path, sheet_name, col_index)
                mr_chart(file_path, sheet_name, col_index)
                normal_probability_plot(file_path, sheet_name, col_index)
            messagebox.showinfo("Success", "Plots generated for selected columns.")
            if messagebox.askyesno("Exit", "Do you want to exit the application?"):
                self.exit_app()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExcelColumnApp(root)
    def on_closing():
        app.exit_app()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()