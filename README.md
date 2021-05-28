# <div align="center"> S🔧MM⚙️D </div>
### <div align="center"> A Framework to Modify Simulations Online </div>


<p align="center">
    <a href="https://choosealicense.com/licenses/mit/">
        <img alt="license" src="https://img.shields.io/github/license/moritztaylor/simmod"/>
    </a>
    <a href="https://pypi.org/project/disent">
        <img alt="python version" src="https://img.shields.io/badge/python-3.8-brightgreen"/>
    </a>
</p>

A simple framework to change visual and physical parameters of various simulations without reloading the simulation and through a standardized interface.

Changing parameters online is especially useful for domain randomization and curriculum learning in robotics research. Thus several algorithms for domain randomization are directly included (Uniform Domain Randomization, ...).

Parameters can be changed directly with the simulation attributes or if applicable over corresponding OpenAI Gym wrapper.

Currently supported simulations:
- Mujoco

Furthermore, online modification of built-in Python variables of the simulation class is also supported.

## Installation
Clone the repo and cd into it:
```bash
git clone https://github.com/MoritzTaylor/simmod.git
cd simulation-modification-framework
```

Install simmod package:
```bash
pip install -e .
```

## Structure
Each simulation can be changed through a handful of modifiers. Every modifier changes a specific category of parameters of the simulation (i.e. material of the object, body properties, etc.). Modifiers can be configured with configurations which specify the parameters/objects to change. The elements of the configurations are stored in `parameterization` objects.

## Usage
Usage is simple and can be performed in 3 steps which is generally structured as follows:
```python
sim = Simulation(...)  # Initialize the simulation
mod = ModifierClass(sim)  # Initialize the modifier
mod.set_property(object_name, new_value)  # Change the property to the new value
sim.step()  # Usually the property gets changed in the next simulation step
```

Functions like `set_property` are individual for each modifier and can be retrieved by the modifier property `standard_setters`. The simulation objects which can be changed by the modifier can be retrieved similarly using the modifier property `names` .

Using a domain randomization algorithm in addition is just as simple:
```python
alg = Algorithm(mod1, mod2, mod3, ...)  # An algorithm gets one or more modifiers as input
alg.step()  # Change the parameters defined in the modifiers in one step
```

An example looks like this:
```python
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from simmod import load_yaml
from simmod.algorithms import UniformDomainRandomization
from simmod.modification.mujoco import MujocoBodyModifier

# Create and load the simulation
model = load_model_from_xml(MODEL_STRING)
sim = MjSim(model)
sim.step()

# Define modifier and algorithm for randomization
config = load_yaml('./examples/assets/algorithm_example.yaml')
mod_body = MujocoBodyModifier(sim=sim, config=config)
alg = UniformDomainRandomization(mod_body)

# Run algorithm and simulation
alg.step()
sim.step()
```

More examples can be found in the _examples_ directory.

## Configuration
As you can see in the example above, all modifiers can be configured with configurations so that specific objects with individual upper and lower value bounds can be defined. That is especially useful when using domain randomization algorithms to define their sampling range.

Configurations are stored as dictionaries and can be externally loaded from yaml files.

### Modifiers
Each modifier is built on a configuration. If no configuration is given it is taken from a standard configuration file in [simmod/modification/data/](./simmod/modification/data/) which
usually defines the object parameter bounds at infinity or zero. This might make the simulation unstable. Therefore, it is recommended to define a configuration for each modifier individual to your specific application. 
Each property of the modifier (i.e. `mass`) holds the objects that should be changed. For each of these objects an upper or lower bound must be given in a tuple (one tuple for each dimension).

Example:
```yaml
---
options:
  execution: 'RESET'
mass:
  pole:
    - [0.018, 0.03]  # [lower bound, upper bound]
  arm:
    - [0.05175, 0.08625]
```
This configuration only changes the mass of the objects named `pole` and `arm` in the simulation when using a modifier which changes the property `mass` (usually modifier which change bodies). The `options` category defines specific options of the modifier and the algorithms. I.e. the attribute `execution` defines when the simulation properties gets changed (i.e. 'RESET', 'BEFORE_STEP' and 'AFTER_STEP').
If a configuration is given, only those objects which are defined in
this configuration are used by the domain randomization algorithms.
If all other objects should be changed in a specific range this can be specified via an additional object called `default`:
```yaml
---
options:
  execution: 'RESET'
mass:
  arm:
    - [0.05175, 0.08625]
  default:
    - [0, .inf] 
    # lower and upper bound of each object except for object with name 'arm'
```
Therefore, it is important that no object in the simulation is named `default` to avoid unexpected behavior.

### Experiments
To run many experiments in one single run, configuration files for modifiers can be combined into a single larger configuration file:

```yaml
---
experiment-1:
    MujocoBodyModifier:
        options:
            execution: 'RESET'
        mass:
            pole:
                - [0.018, 0.03]

experiment-2:
    MujocoBodyModifier:
        options:
            execution: 'RESET'
        mass:
            pole:
                - [0.018, 0.03]
            arm:
                - [0.05175, 0.08625]
    MujocoJointModifier:
        options:
            execution: 'RESET'
        damping:
            base_motor:
              - [ 0.000373125, 0.000621875 ]
            arm_pole:
              - [ 0.0000748875, 0.0001248125 ]
```

Those experiment configurations can be loaded with special utility functions:
```python
from simmod.utils.experiment_utils import GymExperimentScheduler

exp_scheduler = GymExperimentScheduler()
exp_scheduler.load_experiments("./examples/assets/experiment_example.yaml")

for experiment in iter(experiments):
    modifiers = exp_scheduler.create_modifiers(experiment.configurations, env)
    ...
```

## OpenAI Gym Wrapper
All modifiers and algorithms are combined in OpenAI Gym wrapper. Those can be used in place of the before used algorithm:
```python
import gym
from simmod import load_yaml
from simmod.wrappers import UDRMujocoWrapper
from simmod.modification.mujoco import MujocoBodyModifier

# Create the environment as you would normally do
env = gym.make('InvertedPendulum-v2')

# Define modifier and algorithm for randomization
config = load_yaml('./examples/assets/algorithm_example.yaml')
# env.sim is the Mujoco simulation in the environment class
mod_body = MujocoBodyModifier(sim=env.sim, config=config)
env = UDRMujocoWrapper(env, mod_body)

# Run algorithm and simulation
env.step()
```

## Software used
### MuJoCo
Some of the modifiers use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). To run MuJoCo with Python the mujoco-py package is needed. Instructions on setting up mujoco-py can be found [here](https://github.com/openai/mujoco-py).
