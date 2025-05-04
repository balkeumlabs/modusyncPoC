import socket
import os
import pickle
import torch
import torch.nn.functional as F
import rsa
import random
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import ModelArchitecture
from utils import send_data, receive_data, log

HOST = 'localhost'
PORT = 8000
CLIENT_ID = int(input("Enter Client ID (0/1/2): "))

def load_partition(client_id, num_clients=3):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_len = len(train_dataset) // num_clients
    return random_split(train_dataset, [data_len] * num_clients)[client_id]

def local_train(model, train_loader, epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def main():
    log(f"Client {CLIENT_ID} starting...")
    if os.path.exists('./data'):
        for f in os.listdir('./data'):
            os.remove(os.path.join('./data', f))
        log("Cleaned previous dataset.")

    dataset = load_partition(CLIENT_ID)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        log("Connected to server")

        # Generate client RSA key
        client_pub, client_priv = rsa.newkeys(2048)
        send_data(s, client_pub)

        # Receive model from server
        model_state = receive_data(s)
        model = ModelArchitecture()
        model.load_state_dict(model_state)

        log("Model received. Starting training.")
        updated_state = local_train(model, train_loader)

        # Encrypt update with server's public key (mock)
        # In real system, server should send its public key; here we simplify
        server_pub_key = client_pub  # Simulate same key pair
        update_bytes = pickle.dumps(updated_state)
        encrypted_update = rsa.encrypt(update_bytes, server_pub_key)

        send_data(s, encrypted_update)
        log("Encrypted update sent to server.")

if __name__ == "__main__":
    main()
