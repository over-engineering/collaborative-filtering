import os
from zipfile import ZipFile
from urllib import request
import sys

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

def progressBar(value, endvalue, text, bar_length=20, newline=True):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.write("\t%s" % text)
    sys.stdout.flush()

    if value == endvalue and newline:
        sys.stdout.write("\n")