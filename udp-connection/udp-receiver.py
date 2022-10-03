import socket
import pickle
import struct
'''
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
''' 
information = ("127.0.0.2",5000) 
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind(information)
def listen():
    try:
        
        while True:      
                
            data, addr = sock.recvfrom(8192) # buffer size is 1024 bytes
            header = data[:2]
            data = data[2:]
            print(struct.unpack('h',header))
            print(pickle.loads(data))
    except:
        listen()    
listen()
    
        
    