import os
import pickle

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep

from sagemaker_pipeline_facade import LIB_PATH


class ProcessingParser:
    def __init__(self, root_dir, role, pipeline_session):
        self.root_dir = root_dir
        self.role = role
        self.pipeline_session = pipeline_session

    def parse(self, step):
        step.code_dir = '/opt/ml/processing/input/code'
        step.input_dir = '/opt/ml/processing/input'
        step.output_dir = '/opt/ml/processing/output'

        step_inputs = self.default_inputs(step.code_dir)
        step_inputs += [
            ProcessingInput(
                input_name=item.name if item.name else f'input_{index}',
                source=item.source,
                destination=f'{step.input_dir}/{item.name}'
            )
            for index, item in enumerate(step.inputs)
        ]

        processor = self.get_processor()

        args = processor.run(
            inputs=step_inputs,
            outputs=[
                ProcessingOutput(
                    output_name=item.name if item.name else f'output_{index}',
                    source=f'{step.output_dir}/{item.name}'
                )
                for index, item in enumerate(step.outputs)
            ],
            code=self.export_processing_step(step),
        )

        step.parsed_step = ProcessingStep(
            name=step.name(), step_args=args
        )

        return step.parsed_step

    def default_inputs(self, code_dir):
        default_inputs = [
            ProcessingInput(
                input_name='lib',
                source=LIB_PATH,
                destination=f'{code_dir}/{LIB_PATH.split("/")[-1]}'
            ),
            ProcessingInput(
                input_name='root',
                source=self.root_dir,
                destination=f'{code_dir}/{self.root_dir.split("/")[-1]}'
            )
        ]
        return default_inputs

    def get_processor(self):
        return SKLearnProcessor(
            framework_version='1.2-1',
            instance_type="ml.m5.xlarge",
            instance_count=1,
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

    @staticmethod
    def export_processing_step(step):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        export_dir = f'{current_dir}/export'

        template = os.path.join(current_dir, 'processing_step_template.py')
        content = ''

        with open(template, 'r') as file:
            content = file.read()

        content = content.replace(
            '"<serialized-step>"', str(pickle.dumps(step))
        )

        if not os.path.exists(export_dir):
            os.mkdir(export_dir)

        step_file_path = os.path.join(export_dir, f'{step.name()}.py')

        with open(step_file_path, 'w') as file:
            file.write(content)

        return step_file_path
