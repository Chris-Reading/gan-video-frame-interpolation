import matplotlib.pyplot as plt
import numpy as np

def dataset_plot(train_data, batch_size):
     for idx, batch in enumerate(train_data):
        print(f"Inside for loop, index {idx}")

        frames, label = batch
        for i in range(batch_size):
            currentframe1 = frames[0][i]
            currentframe2 = frames[1][i]
            currentframe3 = frames[2][i]
            currentframe4 = frames[3][i]
            currentframe5 = frames[4][i]
            currentframe6 = frames[5][i]
            currentframe7 = frames[6][i]

            plt.figure()
            plt.axis("off")
            plt.title("Test")
            plt.subplot(1, 7, 1)
            plt.axis("off")
            plt.imshow(np.transpose(currentframe1, (1, 2, 0)))
            plt.title("1")

            plt.subplot(1, 7, 2)
            plt.axis("off")
            plt.imshow(np.transpose(currentframe2, (1, 2, 0)))
            plt.title("2")

            plt.subplot(1, 7, 3)
            plt.axis("off")
            plt.imshow(np.transpose(currentframe3, (1, 2, 0)))
            plt.title("3")

            plt.subplot(1, 7, 4)
            plt.axis("off")
            plt.imshow(np.transpose(currentframe4, (1, 2, 0)))
            plt.title("4")

            plt.subplot(1, 7, 5)
            plt.axis("off")
            plt.imshow(np.transpose(currentframe5, (1, 2, 0)))
            plt.title("5")

            plt.subplot(1, 7, 6)
            plt.axis("off")
            plt.imshow(np.transpose(currentframe6, (1, 2, 0)))
            plt.title("6")

            plt.subplot(1, 7, 7)
            plt.axis("off")
            plt.imshow(np.transpose(currentframe7, (1, 2, 0)))
            plt.title("7")

            plt.tight_layout()
            plt.show()