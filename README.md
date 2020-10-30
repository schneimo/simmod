# SimMod - A Framework for Simulation Modification

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://gitlab.is.tue.mpg.de/mschneider/simulation-modification-framework.git
    cd simulation-modification-framework
    ```
- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the modifiers use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). To run MuJoCo with Python the mujoco-py package is needed. Instructions on setting up mujoco-py can be found [here](https://github.com/openai/mujoco-py)
