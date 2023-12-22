from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class Param:
    name: str = None
    source: str = None
    destination: str = None


class Step(ABC):

    code_dir = '/opt/ml/processing/input/code'
    input_dir = '/opt/ml/processing/input'
    output_dir = '/opt/ml/processing/output'

    def __init__(self):
        self.script_path = None

    @abstractmethod
    def execute(self):
        pass

    def name(self):
        name = type(self).__name__
        return name[:-4] if name.endswith('Step') else name

    def inputs(self):
        return []

    def outputs(self):
        return []

    def read_input_csv(self, input_name, **kwargs):
        source = [i.source for i in self.inputs() if i.name == input_name][0]
        file_name = source.split('/')[-1]
        return pd.read_csv(
            f"{self.input_dir}/{input_name}/{file_name}",
            **kwargs
        )

    def write_output_csv(self, output_name, df, output_file_name=None, **kwargs):
        if not output_file_name:
            output_file_name = f'{output_name}.csv'
        pd.DataFrame(df).to_csv(
            f"{self.output_dir}/{output_name}/{output_file_name}",
            **kwargs
        )
