from gymnasium.envs.registration import register

register(id="simplemoenv", entry_point="envs.collaborative.collaborative:TestSimpleMOEnv", nondeterministic=False)
