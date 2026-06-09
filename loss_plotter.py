import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def get_loss_data(file_path, avrg_window=8):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            loss_values = [float(line.strip()) for line in lines if line.strip()]
        
        if len(loss_values) > avrg_window:
            return np.convolve(loss_values, np.ones(avrg_window)/avrg_window, mode='valid')
        return loss_values
    except (FileNotFoundError, ValueError):
        return []

def plot_live_loss(file_path, avrg_window=200, interval=2000):
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], label="Loss")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Live Training Loss History")
    ax.legend()
    ax.grid()

    def update(frame):
        data = get_loss_data(file_path, avrg_window)
        if len(data) > 0:
            line.set_data(range(len(data)), data)
            ax.relim()
            ax.autoscale_view()
        return line,

    # interval is in milliseconds
    ani = FuncAnimation(fig, update, interval=interval, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    plot_live_loss("loss_history.txt")