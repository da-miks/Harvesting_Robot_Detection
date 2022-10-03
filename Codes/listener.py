import socket
import detect
'''
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
''' 
information = ("127.0.0.1",5000) 
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind(information)

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    if data == b'1':
        print("Erkennung wird jetzt ausgeführt")
        