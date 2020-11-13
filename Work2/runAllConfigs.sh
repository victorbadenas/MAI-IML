#!bin/bash

configFiles=$(find configs/*);

for configFile in $configFiles; do
    echo $configFile;
    python -u main.py -c $configFile
done;
