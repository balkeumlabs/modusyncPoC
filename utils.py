import pickle
import datetime

def send_data(sock, data):
    serialized = pickle.dumps(data)
    sock.sendall(len(serialized).to_bytes(4, 'big') + serialized)

def receive_data(sock):
    length = int.from_bytes(sock.recv(4), 'big')
    data = b''
    while len(data) < length:
        data += sock.recv(length - len(data))
    return pickle.loads(data)

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
