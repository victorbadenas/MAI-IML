# CLUSTERING EXERCISE

Repository with the necessary code to run the assignment for IML

## Complete run

The following command will run `main.py` with each conf file in the configs folder.
```bash
bash runAllConfigs.sh
```

## Execution instructions

The repository has the following structure:

```
root
│   main.py
│   ReadMe.md
│
├───configs
│       adult.json
│       pen-based.json
│       vote.json
│
├───datasets
│       adult.arff
│       pen-based.arff
│       vote.arff
│
├───doc
├───results
├───scripts
│
└───src
    │   dataset.py
    │   kmeans.py
    │   pca.py
    |   utils.py
    |   visualize.py
    |   __init__.py
    │
    └───metrics
            clusterMappingMetric.py
            purity.py
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

This main script will run the pca algorithm and the phases results and it will store the confusion matrixes and the metrics extracted as well as the plots in the `resultsDir` directory declared in the json file. It will also show the values needed for the delivery.
