from gymnasium.envs.registration import register


register(id="custom-deep-sea-treasure", entry_point="envs.image_state_deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure", max_episode_steps=100)
