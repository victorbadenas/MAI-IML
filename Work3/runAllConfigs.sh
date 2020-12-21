#!bin/bash

configFiles=$(find configs/*.json);

for configFile in $configFiles; do
    echo ===================== $configFile =====================
    python -u main.py -c $configFile
done;
