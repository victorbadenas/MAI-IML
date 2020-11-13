import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def scatter3D(data, show=False, path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10)
        plt.tight_layout()
        if path is not None:
            plt.savefig(path)
        if show:
            plt.show()
        else:
            return fig, ax

    @staticmethod
    def labeledScatter3D(data, labels, show=False, path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for label in set(labels):
            subData = data[labels == label]
            ax.scatter(subData[:, 0], subData[:, 1], subData[:, 2], s=10, label=label)
            ax.view_init(30, 185)
        plt.legend()
        plt.tight_layout()
        if path is not None:
            plt.savefig(path)
        if show:
            plt.show()
        else:
            return fig, ax
