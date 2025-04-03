from gymnasium.envs.registration import register

register(id="mo-collaborative-env", entry_point="envs.collaborative.collaborative:TestSimpleMOEnv", nondeterministic=False)
