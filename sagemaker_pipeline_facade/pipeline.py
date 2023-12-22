import os
import pickle

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline
from sagemaker.workflow.steps import ProcessingStep

from sagemaker_pipeline_facade import LIB_PATH
from sagemaker_pipeline_facade.step import Step


class Pipeline:

    def __init__(self, name, root_dir, role, pipeline_session):
        print(f'Running script: {os.getcwd()}')

        self.name = name
        self.root_path = root_dir
        self.role = role
        self.pipeline_session = pipeline_session

        self.steps = []
        self.processor = None

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.export_dir = f'{self.current_dir}/export'

    def execute(self):
        sagemaker_pipeline = SageMakerPipeline(
            name='AbaloneNew',
            steps=self.steps,
        )

        sagemaker_pipeline.upsert(role_arn=self.role)
        execution = sagemaker_pipeline.start()
        print(execution)

    def get_processor(self):
        if not self.processor:
            self.processor = SKLearnProcessor(
                framework_version='1.2-1',
                instance_type="ml.m5.xlarge",
                instance_count=1,
                base_job_name="sklearn-abalone-process",
                role=self.role,
                sagemaker_session=self.pipeline_session,
            )
        return self.processor

    def add_processing_step(self, step: Step):
        default_inputs = [
            ProcessingInput(
                input_name='lib',
                source=LIB_PATH,
                destination=f'{step.code_dir}/{LIB_PATH.split("/")[-1]}'
            ),
            ProcessingInput(
                input_name='root',
                source=self.root_path,
                destination=f'{step.code_dir}/{self.root_path.split("/")[-1]}'
            )
        ]

        step_inputs = [
            ProcessingInput(
                input_name=item.name if item.name else f'input_{index}',
                source=item.source,
                destination=f'{step.input_dir}/{item.name}'
            )
            for index, item in enumerate(step.inputs())
        ]

        processor = self.get_processor()
        args = processor.run(
            inputs=default_inputs + step_inputs,
            outputs=[
                ProcessingOutput(
                    output_name=item.name if item.name else f'output_{index}',
                    source=f'{step.output_dir}/{item.name}'
                )
                for index, item in enumerate(step.outputs())
            ],
            code=self.export_processing_step(step),
        )

        self.steps.append(
            ProcessingStep(
                name=step.name(), step_args=args
            )
        )

    def export_processing_step(self, step):
        template = os.path.join(self.current_dir, 'processing_step_template.py')
        content = ''

        with open(template, 'r') as file:
            content = file.read()

        content = content.replace('"<serialized-step>"', str(pickle.dumps(step)))

        if not os.path.exists(self.export_dir):
            os.mkdir(self.export_dir)

        step_file_path = os.path.join(self.export_dir, f'{step.name()}.py')

        with open(step_file_path, 'w') as file:
            file.write(content)

        return step_file_path
