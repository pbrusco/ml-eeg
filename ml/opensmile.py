import arff
import time
import os
from . import system


def call_script(temp_folder, config, filename):
    timestamp = time.time()
    temp_output = "{}/{}{}.arff".format(temp_folder, timestamp, os.path.basename(filename))
    data = {}
    system.run_command("SMILExtract -C {} -I {} -l 0 -arffoutput {} -appendarff 0".format(config, filename, temp_output))
    data = arff.load(open(temp_output, 'rb'))
    system.rm(temp_output)
    return data
