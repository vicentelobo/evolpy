# EvolPy

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

EvolPy is an open-source Python library to use evolutionary algorithms for optimization problems. It seeks to be a simple, but powerful library allowing rapid testing.

## Installation

You can install EvolPy using pip:

```bash
  pip3 install git+https://github.com/vicentelobo/evolpy@main
```

If you wish to build from sources, download or clone the repository and type

```bash
  python3 setup.py install
```

If you do not have root privileges, you can install EvolPy for only you (the user!) with this line instead:

```bash
  python3 setup.py install --user
```

Do not forget to install the requirements using the `requirements.txt` file located in the EvolPy root folder:

```bash
  pip3 install -r requirements.txt
```

## Usage

Check the full documentation on our [Read the Docs](https://evolpy.readthedocs.io/en/latest/)

```python
import numpy as np
from collections import OrderedDict

from evolpy import GP

def fitness(individual):
  return np.square(individual).sum() # f(x) = x**2

parameters = OrderedDict([
               ('x0', (-100, 100)),
               ('x1', (-100, 100))
             ])

evolver = GP(fitness=fitness, parameters=parameters, populationSize=10, maxGen=25)
best_individual, fitness_history = evolver.run()
print(best_individual) # Desirable output: {'gene': {'x0': 0.0}, 'fitness': array([0.])}

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[GPL v3](https://www.gnu.org/licenses/gpl-3.0)