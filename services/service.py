import requests

url = 'http://75.119.149.11:30000/Auth/Login'
urlProgram = 'http://75.119.149.11:30002/Atencion/AgregarProgramacion'


def login(data):
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response
    else:
        return None


def programar(headers, data):
    response = requests.post(urlProgram, headers=headers, json=data)
    return response