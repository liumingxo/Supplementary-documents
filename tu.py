# Load necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
data = pd.read_excel("D:/Google/sj_augmented.xlsx", sheet_name=0)

# Define nature color palette
nature_colors = sns.color_palette("husl", 8)

# Convert data to long format for easy plotting
data_long = data.melt(var_name="Variable", value_name="Value")

# Set custom theme
sns.set_theme(style="whitegrid", font_scale=1.2)

# Function to create violin and boxplots
def plot_violin_box(data, variables, title, palette=nature_colors):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Variable", y="Value", data=data[data["Variable"].isin(variables)],
                   inner=None, palette=palette, linewidth=1.2)
    sns.boxplot(x="Variable", y="Value", data=data[data["Variable"].isin(variables)],
                width=0.15, boxprops={"zorder": 2, "facecolor": "none", "edgecolor": "black"},
                showfliers=False)
    plt.title(title, fontsize=16, weight="bold")
    plt.xlabel("Variable", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.show()

# Plot 1: C, H, O, N
plot_violin_box(data_long, ["C", "H", "O", "N"], "Distribution of C, H, O, N")

# Plot 2: K, P, M, Ash
plot_violin_box(data_long, ["K", "P", "M", "Ash"], "Distribution of K, P, M, Ash")

# Plot 3: FC, LT, TR
plot_violin_box(data_long, ["FC", "LT", "TR"], "Distribution of FC, LT, TR")

# Plot 4: T
plot_violin_box(data_long, ["T"], "Distribution of T")

# Plot 5: BY, C, H, O
plot_violin_box(data_long, ["BY", "C", "H", "O"], "Distribution of BY, C, H, O")

# Plot 6: N, P, K
plot_violin_box(data_long, ["N", "P", "K"], "Distribution of N, P, K")
