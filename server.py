import socket, threading, pickle, torch
from model import ModelArchitecture, initialize_weights
from utils import receive_data, send_data, log
import time
from cryptography.hazmat.primitives.asymmetric import rsa as crypto_rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from utils_crypto import aes_decrypt, rsa_decrypt_key
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

HOST = 'localhost'
PORT = 8000
NUM_CLIENTS = 3
NUM_ROUNDS = 10

clients = []

# --- Evaluate model ---
def evaluate(model):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    return acc


def client_handler(conn, addr, cid):
    clients.append((cid, conn, addr))

def listen_for_clients():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        log("Server listening for clients...")
        while len(clients) < NUM_CLIENTS:
            conn, addr = s.accept()
            cid = len(clients)
            log(f"Client {cid} connected from {addr}")
            threading.Thread(target=client_handler, args=(conn, addr, cid)).start()
        log("All clients connected.")

def federated_round(round_num, model):
    log(f"=== ROUND {round_num} STARTED ===")

    private_key = crypto_rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
    )
    public_key = private_key.public_key()
    public_key_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

    # Broadcast public key and model
    for cid, conn, _ in clients:
        payload = {'round': round_num, 'public_key': public_key_pem}
        if round_num == 0:
            payload['model_arch'] = model
        else:
            payload['model_state'] = model.state_dict()
        send_data(conn, payload)

    # Collect encrypted updates
    updates = []
    for cid, conn, _ in clients:
        payload = receive_data(conn)
        rsa_encrypted_key = payload['rsa_key']
        iv = payload['iv']
        encrypted_update = payload['data']
        aes_key = rsa_decrypt_key(rsa_encrypted_key, private_key)
        update_bytes = aes_decrypt(encrypted_update, aes_key, iv)
        
        update = pickle.loads(update_bytes)
        
        updates.append(update)
        log(f"Update received from Client {cid}")

    # Aggregate
    global_state = updates[0]
    for key in global_state:
        global_state[key] = torch.stack([u[key] for u in updates], 0).mean(0)

    model.load_state_dict(global_state)
    # Run evaluation
    accuracy = evaluate(model)
    log(f"=== ROUND {round_num} COMPLETED ===")
    log(f"Global model accuracy on test set: {accuracy:.2f}%")
    return model

def main():
    listen_for_clients()
    model = ModelArchitecture()
    model.apply(initialize_weights)
    log("Genesis model initialized.")

    for r in range(NUM_ROUNDS):
        model = federated_round(r, model)
        time.sleep(2)

    log("Training complete. Final model ready.")

if __name__ == "__main__":
    main()
