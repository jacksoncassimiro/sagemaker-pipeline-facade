from abc import abstractmethod
from typing import List

import pandas as pd
from sagemaker_pipeline_facade import Param, FacadeStep


class ProcessingFacadeStep(FacadeStep):

    def __init__(
            self,
            inputs: List[Param] = None,
            outputs: List[Param] = None,
    ):
        self.script_path = None
        self.parsed_step = None
        self.code_dir = None
        self.input_dir = None
        self.output_dir = None

        self.inputs = inputs
        self.outputs = outputs

    @abstractmethod
    def execute(self):
        pass

    def get_output_value_as_param(self, name, output_name, content_type=None):
        source = (
            self.parsed_step.properties.ProcessingOutputConfig
            .Outputs[output_name].S3Output.S3Uri
        )
        return Param(
            name=name,
            source=source,
            content_type=content_type
        )

    def read_input_csv(self, input_name, file_name, **kwargs):
        return pd.read_csv(
            f"{self.input_dir}/{input_name}/{file_name}",
            **kwargs
        )

    def write_output_csv(
            self, output_name, df, output_file_name=None, **kwargs
    ):
        if not output_file_name:
            output_file_name = f'{output_name}.csv'
        pd.DataFrame(df).to_csv(
            f"{self.output_dir}/{output_name}/{output_file_name}",
            **kwargs
        )
