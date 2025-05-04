import socket
import threading
import pickle
import torch
import rsa
from model import ModelArchitecture, initialize_weights
from utils import receive_data, send_data, log

HOST = 'localhost'
PORT = 8000
NUM_CLIENTS = 3

model = ModelArchitecture()
model.apply(initialize_weights)

clients = []
public_keys = {}
updates = []

def handle_client(conn, addr, client_id, private_key):
    log(f"Client {client_id} connected from {addr}")
    
    pubkey = receive_data(conn)
    public_keys[client_id] = pubkey

    # Send initial model
    send_data(conn, model.state_dict())

    # Receive encrypted update
    encrypted_update = receive_data(conn)
    update_bytes = rsa.decrypt(encrypted_update, private_key)
    update = pickle.loads(update_bytes)
    updates.append(update)
    
    log(f"Received update from Client {client_id}")

    if len(updates) == NUM_CLIENTS:
        aggregate_updates()

def aggregate_updates():
    global model
    new_state = model.state_dict()
    for k in new_state:
        new_state[k] = torch.stack([update[k] for update in updates], 0).mean(0)
    model.load_state_dict(new_state)
    log("Model aggregated successfully.")

def main():
    log("Loading server...")

    # Generate asymmetric key pair
    public_key, private_key = rsa.newkeys(2048)
    log("RSA key pair generated.")

    # Start server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        log(f"Server listening on {HOST}:{PORT}")

        for client_id in range(NUM_CLIENTS):
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr, client_id, private_key))
            thread.start()

if __name__ == "__main__":
    main()
