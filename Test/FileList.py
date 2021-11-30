import os
import datetime


path = r"C:\Users\WIN10-01\Documents\Python"

def search_file(directory):
    print(f"== search_file({directory})")

    list_files = os.listdir(directory)
    
    dirs = [f for f in list_files if os.path.isdir(os.path.join(directory, f))]
    for target in dirs:
        print(f"dir --> {target}")
    files = [f for f in list_files if os.path.isfile(os.path.join(directory, f))]
    for target in files:
        full_name = os.path.join(directory, target)
        stat = os.stat(os.path.abspath(full_name))
        timestamp = datetime.datetime.fromtimestamp(stat.st_mtime)
        print(f" {target} {timestamp:%Y/%m/%d %H:%M:%S}")

    for target_directory in dirs:
        search_file(os.path.join(directory, target_directory))

search_file(path)
