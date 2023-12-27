import os
import pickle

from sagemaker.processing import ProcessingInput, ProcessingOutput, \
    ScriptProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep
from sagemaker_pipeline_facade import (
    LIB_PATH, DEFAULT_INSTANCE_TYPE, DEFAULT_INSTANCE_COUNT, PropertyParam
)
from sagemaker_pipeline_facade.processing_step import ProcessingFacadeStep


class ProcessingStepParser:
    def __init__(self, root_dir, image_uri, role, pipeline_session):
        self.root_dir = root_dir
        self.image_uri = image_uri
        self.role = role
        self.pipeline_session = pipeline_session

    def parse(self, step: ProcessingFacadeStep):
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

        property_files = [
            PropertyFile(
                name=f'{item.name}_property_file',
                output_name=item.name,
                path=f'{item.name}.json'
            )
            for item in step.outputs if isinstance(item, PropertyParam)
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
            name=step.name(),
            step_args=args,
            property_files=property_files
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
        return ScriptProcessor(
            image_uri=self.image_uri,
            command=['python3'],
            instance_type=DEFAULT_INSTANCE_TYPE,
            instance_count=DEFAULT_INSTANCE_COUNT,
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

    def export_processing_step(self, step: ProcessingFacadeStep):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        export_dir = f'{current_dir}/export'

        template = os.path.join(current_dir, 'processing_step_template.py')
        content = ''

        with open(template, 'r') as file:
            content = file.read()

        step_copied = self.copy(step)
        for param in step_copied.inputs:
            param.source = None

        content = content.replace(
            '"<serialized-step>"', str(pickle.dumps(step_copied))
        )

        if not os.path.exists(export_dir):
            os.mkdir(export_dir)

        step_file_path = os.path.join(export_dir, f'{step.name()}.py')

        with open(step_file_path, 'w') as file:
            file.write(content)

        return step_file_path

    @staticmethod
    def copy(arg):
        return pickle.loads(pickle.dumps(arg))
