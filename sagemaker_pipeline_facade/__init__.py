import os
from abc import ABC
from dataclasses import dataclass


@dataclass
class Param:
    name: str = None
    source: object = None
    destination: str = None
    content_type: str = None


@dataclass
class PropertyParam(Param):
    pass


class FacadeStep(ABC):

    def __init__(self):
        self.parsed_step = None

    def name(self):
        name = type(self).__name__
        return name[:-10] if name.endswith('FacadeStep') else name


LIB_PATH: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INSTANCE_TYPE: str = 'ml.m5.xlarge'
DEFAULT_INSTANCE_COUNT: int = 1
