import socket
import struct
'''
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
MESSAGE = b"Hello, World!"

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)
   
sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
'''

information = ("127.0.0.1",5000)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.sendto(struct.pack('h',5),information)
