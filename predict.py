import torch
from network import NeuralNetwork
from dataset import test_data

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


def predict(index):
    model.eval()
    x, y = test_data[index][0], test_data[index][1] # x is the image, y is the label
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)

        print(f'Predicted scores: {pred[0]}')

        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Most likely class - predicted: "{predicted}", Actual: "{actual}"')
