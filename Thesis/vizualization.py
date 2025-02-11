# vizualization.py

import matplotlib.pyplot as plt

def plot_gantt(schedule, title):
    """
    Plot a Gantt chart for the given schedule.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks = []
    yticklabels = []
    for i, task in enumerate(schedule):
        ax.broken_barh([(task["start"], task["duration"])],
                       (i * 10, 9),
                       facecolors='tab:blue')
        yticks.append(i * 10 + 5)
        yticklabels.append(f'Task {task["task_id"]}: {task["task_name"]}\n(Workers: {task["workers"]})')
        ax.text(task["start"] + task["duration"] / 2, i * 10 + 5,
                f'{task["start"]:.1f}-{task["finish"]:.1f}',
                ha='center', va='center', color='white', fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Tasks")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
