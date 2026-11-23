"""Model adapters: map ROS observations to a policy server's request format and back."""
from . import openpi_droid

ADAPTERS = {"openpi_droid": openpi_droid}


def get(name: str):
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter '{name}', known: {sorted(ADAPTERS)}")
    return ADAPTERS[name]
