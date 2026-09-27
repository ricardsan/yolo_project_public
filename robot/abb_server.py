import socket

ROBOT_HOST = "192.168.125.1"
ROBOT_PORT = 5004

def send_message(msg: str) -> str:
    parts = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            sock.connect((ROBOT_HOST, ROBOT_PORT))
            sock.sendall(msg.encode("utf-8"))

            while True:
                p = sock.recv(4096)
                if not p:
                    break
                parts.append(p)

    except Exception as e:
        print("[ERROR]", e)

    return b"".join(parts).decode()

text = "test"

print("Sending:", text)

response = send_message(text)

print("Robot replied:", response)