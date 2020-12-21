# LAZY LEARNING EXERCISE

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
│   runAllConfigs.sh
│
├───configs
│       hypothyroid.json
│       splice.json
│
├───10f datasets
│       <folders with folds>
│
├───doc
├───results
├───scripts
│
└───src
    │   dataset.py
    │   knn.py
    │   metrics.py
    │   reductionKnn.py
    │   T_test.py
    |   utils.py
    |   __init__.py
    │
    └───reductions
            ___init__.py
            enn.py
            fcnn.py
            ib.py


```

the configs folder contains the json required to run the main.py script. The json has the following structure:

```json
{
    "dataset": "hypothyroid",
    "knnparameters": {
        "metric": "cosine",
        "n_neighbors": 5,
        "p": 2,
        "voting": "majority",
        "weights": "mutual_info"
    },
    "reduction": ["ib2",
        "fcnn",
        "enn"
    ]
}
```

With that JSON, the `main.py` file can be called using the following command:

```bash
python main.py -c <path_to_json>
```

This main script will run the kNNAlgorithm and the reductionkNNAlgorithm with the parameters specified. If no reduction is desired, just delete the reduction field in the json config. The json must include the dataset (only the name) to which it makes reference to and the knn parameters. If no parameters are passed, it will run it with the default parameters.
