import socket
import pickle
from cv2 import add

HOST = "127.0.0.1"
PORT = 5000

def server():

    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    s.bind((HOST,PORT))

    s.listen()

    c,addr = s.accept()

    print("Connection From: ",str(addr))

    msg = c.recv(4096)

    print(pickle.loads(msg))

    s.close()
    
server()