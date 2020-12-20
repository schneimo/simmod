from simmod.utils.load_utils import load_yaml
from simmod.utils.experiment_utils import GymExperimentScheduler

test = load_yaml('test.yaml')

exp_scheduler = GymExperimentScheduler()
exp_scheduler.load_experiments(config=test)

for _, experiment in enumerate(iter(exp_scheduler)):
    print(experiment.name)
    print(exp_scheduler.create_modifiers(experiment.configurations, env=None))
    print(exp_scheduler.create_wrapped_env(experiment.configurations, env=5))
    print(experiment.configurations)