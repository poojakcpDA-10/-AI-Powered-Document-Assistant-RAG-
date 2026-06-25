import socket

hosts = [
    "huggingface.co",
    "api-inference.huggingface.co"
]

for host in hosts:
    try:
        print(host, "->", socket.gethostbyname(host))
    except Exception as e:
        print(host, "-> ERROR:", e)