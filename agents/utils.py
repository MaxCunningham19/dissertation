from .democratic_dwn import DemocraticDWL
from .AbstractAgent import AbstractAgent
from .dwn import DWL
from .scaled_democratic import ScaledDemocraticDQN
from .democratic import DemocraticDQN
from action_scalarization import get_scalarization_method
from .democratic_dwn.SelectedPolicy import get_selected_policy

agent_names = [
    "scaled_democratic",
    "dwl",
    "democratic",
    "democratic_dwl",
    "democratic_dwl_max_action",
    "democratic_dwl_random_max_action",
    "democratic_dwl_max_action_max_q_value",
]


def get_agent(agent_name: str, **kwargs) -> AbstractAgent:
    if kwargs.get("scalarization", None) is not None:
        kwargs["scalarization"] = get_scalarization_method(kwargs["scalarization"])
    if agent_name == "scaled_democratic":
        return ScaledDemocraticDQN(**kwargs)
    elif agent_name == "dwl":
        return DWL(**kwargs)
    elif agent_name == "democratic":
        return DemocraticDQN(**kwargs)
    elif agent_name.startswith("democratic_dwl"):
        return get_democratic_dwl_agent(agent_name, **kwargs)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}. Possible values are: {agent_names}")


def get_democratic_dwl_agent(agent_name: str, **kwargs) -> AbstractAgent:
    selcted_policy_method = "none"
    if agent_name == "democratic_dwl":
        selcted_policy_method = "none"
    elif agent_name == "democratic_dwl_max_action":
        selcted_policy_method = "selected_action_is_max_action"
    elif agent_name == "democratic_dwl_max_w_value":
        selcted_policy_method = "max_w_value"
    elif agent_name == "democratic_dwl_max_action_max_q_value":
        selcted_policy_method = "max_action_value"
    else:
        raise ValueError(f"Invalid agent name: {agent_name}. Possible values are: {agent_names}")

    kwargs["selected_policy"] = get_selected_policy(selcted_policy_method)
    return DemocraticDWL(**kwargs)
