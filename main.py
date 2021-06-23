from .api.config import Config
from ._calculator import _Calculator


def run(config: Config) -> None:
    _Calculator(config)
