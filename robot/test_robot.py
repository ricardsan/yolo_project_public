import socket

HOST = "192.168.125.1"
PORT = 5004

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"test")
    data = s.recv(1024)

print("Robot replied:", data.decode())