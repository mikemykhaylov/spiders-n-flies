import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data from the experiment
policies = ['Base Policy', 'Regular Rollout', 'Multiagent Rollout']
moves = [36.92, 22.31, 22.43]  # Average moves
times = [0.128, 6.867, 2.618]  # Average time in ms

# Create DataFrames for plotting
moves_df = pd.DataFrame({'Policy': policies, 'Average Moves': moves})

times_df = pd.DataFrame({'Policy': policies, 'Time (ms)': times})

# Set style
sns.set_style("whitegrid")

# Plot and save moves comparison
plt.figure(figsize=(6, 5))
moves_plot = sns.barplot(data=moves_df, x='Policy', y='Average Moves', hue='Policy', palette='deep')
plt.title('Average Moves by Policy')
plt.xticks(rotation=45)
plt.ylabel('Number of Moves')

# Add value labels on bars
for i, v in enumerate(moves):
    moves_plot.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('moves.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot and save time comparison
plt.figure(figsize=(6, 5))
times_plot = sns.barplot(data=times_df, x='Policy', y='Time (ms)', hue='Policy', palette='deep')
plt.title('Average Computation Time by Policy')
plt.xticks(rotation=45)
plt.ylabel('Time (milliseconds)')

# Add value labels on bars
for i, v in enumerate(times):
    times_plot.text(i, v, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('times.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graphs have been generated and saved as 'moves.png' and 'times.png'")
