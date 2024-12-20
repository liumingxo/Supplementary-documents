import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = r"D:\\Google\\sj_augmented.xlsx"
data = pd.read_excel(file_path)

# Define the column groups
column_groups = {
    "Group1": ["C", "H", "O", "N"],
    "Group2": ["K", "P", "M", "Ash"],
    "Group3": ["FC", "LT", "TR"],
    "Group4": ["T"],
    "Group5": ["BY", "C", "H", "O"],
    "Group6": ["N", "P", "K"]
}


# Check for negative values and correct them
def correct_negative_values(data):
    for col in data.columns:
        if data[col].dtype in ["float64", "int64"]:  # Ensure it's a numeric column
            negative_count = (data[col] < 0).sum()
            if negative_count > 0:
                print(f"Column '{col}' contains {negative_count} negative values. Setting them to 0.")
                data[col] = data[col].apply(lambda x: max(x, 0))
    return data


data = correct_negative_values(data)

# Plot style
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Generate violin plots
for group_name, columns in column_groups.items():
    plt.figure(figsize=(10, 6))

    # Melt the data for seaborn violin plot
    melted_data = data[columns].melt(var_name="Variable", value_name="Value")

    # Generate violin plot
    sns.violinplot(x="Variable", y="Value", data=melted_data, inner="box", palette="viridis", linewidth=1.5)

    # Beautify plot
    plt.title(f"Violin Plot for {group_name}", fontsize=14, fontweight='bold')
    plt.ylabel("Relative Contribution (%)", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plot_file = f"D:\\Google\\{group_name}_violin_plot.png"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved plot: {plot_file}")

    plt.show()
