import socket, pickle, torch, rsa, os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import nn, optim
from model import ModelArchitecture
from utils import receive_data, send_data, log
import torch.nn.functional as F

HOST = 'localhost'
PORT = 8000
NUM_CLIENTS = 3
CLIENT_ID = int(input("Enter Client ID (0/1/2): "))

def load_partition(client_id):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    length = len(train_dataset) // NUM_CLIENTS
    partitions = random_split(train_dataset, [length]*NUM_CLIENTS)
    return partitions[client_id]

def train_local(model, dataloader, epochs=1):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for x, y in dataloader:
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
    return model.state_dict()

def main():
    if os.path.exists('./data'):
        log("Using existing MNIST data.")

    partition = load_partition(CLIENT_ID)
    loader = DataLoader(partition, batch_size=32, shuffle=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        log(f"Connected to server as Client {CLIENT_ID}")

        for round_num in range(10):
            payload = receive_data(s)
            public_key = payload['public_key']

            if round_num == 0:
                model = payload['model_arch']
                log(f"Round {round_num}: Genesis model received.")
            else:
                model.load_state_dict(payload['model_state'])
                log(f"Round {round_num}: Model state received.")

            updated_state = train_local(model, loader)

            update_bytes = pickle.dumps(updated_state)
            encrypted = rsa.encrypt(update_bytes, public_key)

            send_data(s, encrypted)
            log(f"Round {round_num}: Update sent.")

if __name__ == "__main__":
    main()
