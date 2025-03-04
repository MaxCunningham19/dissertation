from .AbstractAgent import AbstractAgent
from .DDWN import BaseDDWN
from .dwn import DWL
from .democratic import DemocraticDQN


def get_agent(agent_name: str) -> AbstractAgent:
    if agent_name == "democratic":
        return DemocraticDQN
    elif agent_name == "dwl":
        return DWL
    elif agent_name == "ddwn":
        return BaseDDWN
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")
