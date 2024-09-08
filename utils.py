import numpy as np


def print_batch_accuracy(data, targets, net, batch_size, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def train_printer(
    epoch,
    iter_counter,
    loss_hist,
    test_loss_hist,
    data,
    targets,
    test_data,
    test_targets,
    net,
    batch_size,
):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[-1]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[-1]:.2f}")
    print_batch_accuracy(data, targets, net, batch_size, train=True)
    print_batch_accuracy(test_data, test_targets, net, batch_size, train=False)
    print("\n")
