
import numpy as np
import matplotlib.pyplot as plt

def draw_loss_fig(loss, saving_path):
    loss = np.array(loss)
    steps, loss_mean = loss[:, 0], loss[:, 1]
    plt.clf()
    plt.plot(steps, loss_mean, color='#483D8B')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.grid()
    plt.savefig(saving_path)