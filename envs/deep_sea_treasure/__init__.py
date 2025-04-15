from gymnasium.envs.registration import register

from .deep_sea_treasure import CONCAVE_MAP, MIRRORED_MAP


register(id="mo-deep-sea-treasure-convex-v0", entry_point="envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure", max_episode_steps=100)

register(
    id="mo-deep-sea-treasure-concave-v0",
    entry_point="envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": CONCAVE_MAP},
)

register(
    id="mo-deep-sea-treasure-mirrored-v0",
    entry_point="envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": MIRRORED_MAP},
)

register(
    id="mo-3d-deep-sea-treasure-convex-v0",
    entry_point="envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"invalid_move_objective": True},
)

register(
    id="mo-3d-deep-sea-treasure-concave-v0",
    entry_point="envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": CONCAVE_MAP, "invalid_move_objective": True},
)

register(
    id="mo-3d-deep-sea-treasure-mirrored-v0",
    entry_point="envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": MIRRORED_MAP, "invalid_move_objective": True},
)
