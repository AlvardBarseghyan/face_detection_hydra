import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "r-", label="Training acc")
    plt.plot(epochs, val_acc, "b-", label="Validation acc")
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, "r-", label="Training loss")
    plt.plot(epochs, val_loss, "b-", label="Validation loss")
    plt.legend()
    plt.show()

