import socket, threading, pickle, torch, rsa
from model import ModelArchitecture, initialize_weights
from utils import receive_data, send_data, log
import time

HOST = 'localhost'
PORT = 8000
NUM_CLIENTS = 3
NUM_ROUNDS = 10

clients = []

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

    public_key, private_key = rsa.newkeys(2048)

    # Broadcast public key and model
    for cid, conn, _ in clients:
        payload = {'round': round_num, 'public_key': public_key}
        if round_num == 0:
            payload['model_arch'] = model
        else:
            payload['model_state'] = model.state_dict()
        send_data(conn, payload)

    # Collect encrypted updates
    updates = []
    for cid, conn, _ in clients:
        encrypted_update = receive_data(conn)
        update_bytes = rsa.decrypt(encrypted_update, private_key)
        update = pickle.loads(update_bytes)
        updates.append(update)
        log(f"Update received from Client {cid}")

    # Aggregate
    global_state = updates[0]
    for key in global_state:
        global_state[key] = torch.stack([u[key] for u in updates], 0).mean(0)

    model.load_state_dict(global_state)
    log(f"=== ROUND {round_num} COMPLETED AND AGGREGATED ===")
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
