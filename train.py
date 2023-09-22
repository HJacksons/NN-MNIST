# optimise model and train the model using dataset.py and network.py
import torch
from torch import nn
from network import NeuralNetwork
from dataset import train_dataloader, test_dataloader
import matplotlib.pyplot as plt


# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

losses = []


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # size is the number of samples in the dataset
    losses.clear()
    model.train() # set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # move tensors to the configured device
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad() # clear the gradients of all optimized variables
        loss.backward() # compute gradients of loss wrt all the parameters
        optimizer.step() # perform a single optimization step
        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            losses.append(loss)

    # Plot the loss vs epoch
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # set the model to evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad(): # disable gradient calculation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # move tensors to the configured device
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def run_train():
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")