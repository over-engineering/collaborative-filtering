import os
from zipfile import ZipFile
from urllib import request

def download(url, path):
    try:
        request.urlretrieve(url, path)
    except OSError:
        raise "Error: Failed downloading"

    print("Done")

def unzip(zip_path, extract_path):
    with ZipFile(zip_path, "r") as f:
        print("Extraing %s.zip..." % zip_path.split("/").pop())
        f.extractall(extract_path)
        print("Done")

    os.remove(zip_path)