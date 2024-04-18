import json

def convert(file_name):
    f = open(file_name)
    data = json.load(f)
    print(data)
    f.close()

convert('20160801_102650.json')

