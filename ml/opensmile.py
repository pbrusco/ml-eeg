import arff
import time
import os
from . import system


def call_script(smile_extract_path, temp_folder, config, filename):
    timestamp = time.time()
    temp_output = "{}/{}{}.arff".format(temp_folder, timestamp, os.path.basename(filename))
    data = {}
    if not system.exists(config):
        raise Exception("{} not found".format(config))
    command = "{}/SMILExtract -C {} -I {} -l 0 -arffoutput {} -appendarff 0".format(smile_extract_path, config, filename, temp_output)
    system.run_command(command)
    data = arff.load(open(temp_output))

    system.rm(temp_output)
    return data
