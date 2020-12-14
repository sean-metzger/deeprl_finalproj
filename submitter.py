print('Sko Buffs')
import subprocess
import shlex
import os 
import sys
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

cs285_command = (' ').join(sys.argv[1:])

print('command', cs285_command)
basedir = '/userdata/smetzger/cs285/final_proj_ray/'
times= now.strftime("%d_%m_%Y_%H_%M_%S")
 
filename = basedir + 'runs/' + times + ('_').join(cs285_command.split(' ')) + '.txt'
string = "submit_job -q mind-gpu"
string += " -m 318 -g 4"
string += " -o " + filename
string += ' -n deeprl'
string += ' -x python '
string += basedir + cs285_command

print(string)
cmd = shlex.split(string)
print(cmd)
subprocess.run(cmd, stderr=subprocess.STDOUT)