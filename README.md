# Production rescheduling via explorative reinforcement learning while considering nervousness
<div style="display:flex; align-items: center;">

</div>

RLScheduling is a reinforcement learning PPO algorithm designed to minimize total job assignment costs while considering nervousness.


## setup

```
setup(
    name="RLScheduling",
    version="1.0",
    url="https://github.com/tnals9983/RLScheduling",
    author="Sumin Hwangbo",
    install_requires=[
        "gym == 0.18.3",
        "ray == 1.6.0",
        "ray[rllib] == 1.6.0",
        "pandas == 1.3.5",
        "openpyxl == 3.0.9",
        "tensorflow == 2.9.1",
    ],
    zip_safe=False,
)
```

## Code implementation example

### Train RLScheduling

You can adjust the hyperparameters in the rl_config in the train.py file.

To train the dataset using PPO, please run

```
python train.py
```

### Evaluate the result from checkpoint

After train the data, you can evaluate the results using backsteping method by.

To evalute the result using PPO with trained checkpoint through multiprocessing, please run

```
python parallel_evaluation.py
```

## Data

Datasets related to this article can be found at (http://egon.cheme.cmu.edu/Papers/HarjunkoskiDecompositionCACE-2725.pdf)
