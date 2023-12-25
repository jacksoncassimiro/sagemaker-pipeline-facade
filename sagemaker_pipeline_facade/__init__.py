import os
from abc import ABC
from dataclasses import dataclass


@dataclass
class Param:
    name: str = None
    source: str = None
    destination: str = None
    content_type: str = None


class FacadeStep(ABC):

    def name(self):
        name = type(self).__name__
        return name[:-10] if name.endswith('FacadeStep') else name


LIB_PATH = os.path.dirname(os.path.abspath(__file__))
