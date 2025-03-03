import gym
from gym.envs.registration import register

# Manually register before calling gym.make()
register(id="myenv", entry_point="test_env.env:TestEnv", nondeterministic=False)

register(id="mymoenv", entry_point="test_env.moenv:TestMOEnv", nondeterministic=False)

register(id="simplemoenv", entry_point="test_env.simple_moenv:TestSimpleMOEnv", nondeterministic=False)
