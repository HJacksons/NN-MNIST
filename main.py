from train import run_train
from predict import predict


def run_training():
    run_train()


def run_predict():
    predict(6)

# Option 1: Run the training, 2: Run the prediction


if __name__ == "__main__":
    option = input("Enter 1 to train the model or 2 to predict: ")
    if option == "1":
        run_training()
    elif option == "2":
        run_predict()
    else:
        print("Invalid option")
