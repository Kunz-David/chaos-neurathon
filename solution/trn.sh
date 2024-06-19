#!/usr/bin/env bash

#pid=29587
#
## An entry in /proc means that the process is still running.
#while [ -d "/proc/$pid" ]; do
#    sleep 1
#done

cd /home/team4/chaos_hackaton_2023 || exit

# make sure there is one argument present with path to the yml config
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_yml_config>"
    exit 1
fi

# create identifier for the experiment from the config file name
# and the current time
identifier=$(basename ${1} .yml)_$(date +%Y-%m-%d_%H-%M-%S)

echo "running ${identifier} experiment"

# This script is used to train the model.
/opt/conda/bin/python main.py "${1}" "${identifier}" 2>&1 1>/dev/null