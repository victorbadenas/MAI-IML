# CLUSTERING EXERCISE

Repository with the necessary code to run the assignment for IML

## Execution instructions

The repository has the following structure:

```
root
│   main.py
│   ReadMe.md
│   tree.txt
│
├───configs
│       adult.json
│       pen-based.json
│       vote.json
│
├───datasets
│       *.arff
│
├───doc
├───results
├───scripts
│       3dplotAndMetrics.py
│       datasetPlots.py
│       dbscanScript.py
│       executionTimePlot.py
│
├───src
│   │   dataset.py
│   │   utils.py
│   │   __init__.py
│   │
│   ├───clustering
│   │       bisectingKmeans.py
│   │       fcm.py
│   │       kmeans.py
│   │       kmeansPP.py
│   │       __init__.py
│   │
│   └───metrics
│           clusterMappingMetric.py
│           purity.py
│           __init__.py
│
└───test
        testMain.py
        __init__.py
```

the configs folder contains the json required to run the main.py script. The json has the following structure:

```json
{
    "path": "<relative_path_to_arff_file>",
    "resultsDir": "path_to_results_folder",
    "verbose": <bool>,
    "plotClusters": <bool>,
    "parameters": <nested_objects_with_execution_parameters>
}
```

With that JSON, the `main.py` file can be called using the following command:

```bash
python main.py -c <path_to_json>
```

This main script will run the algorithms for which the json object has parameters for and it will store the confusion matrixes and the metrics extracted for each one of the algorithms in the `resultsDir` directory declared in the json file. It will also show and return the values of the labels as well as show how much time did the algorithms take to run.

## Additional scripts

In the folder scripts, files can be found:

- `3dplotAndMetrics.py` : script to generate PCA representations of the data with color coding for the clusters asigned by each algorithm
- `datasetPlots.py` : script to generate boxplots and scatterplots of the databases
- `dbscanScript.py` : script to find the bes optimal values for the DBScan parameters
- `executionTimePlot.py` : scipt to perform plots and metrics over the average runtime of each algorithm