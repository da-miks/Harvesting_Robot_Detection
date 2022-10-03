
import socket
import pickle
HOST = "127.0.0.1"
PORT = 5000

def client(msg):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    s.connect((HOST,PORT))

    message = pickle.dumps(msg)

    s.sendall(message)

    s.close()


