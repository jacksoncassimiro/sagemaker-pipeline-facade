from datetime import datetime

import pytz
from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline

from sagemaker_pipeline_facade import FacadeStep, Param
from sagemaker_pipeline_facade.processing_step import ProcessingFacadeStep
from sagemaker_pipeline_facade.processing_step_parser import (
    ProcessingStepParser
)
from sagemaker_pipeline_facade.training_step import TrainingFacadeStep
from sagemaker_pipeline_facade.training_step_parser import TrainingStepParser


def get_output_value_as_param(
        step: FacadeStep, name, output_name, content_type=None
):
    source = (
        step.parsed_step.properties.ProcessingOutputConfig
        .Outputs[output_name].S3Output.S3Uri
    )
    return Param(
        name=name,
        source=source,
        content_type=content_type
    )


def get_trained_model_as_param(step: TrainingFacadeStep, name=None):
    if not name:
        name = 'model'
    source = step.parsed_step.properties.ModelArtifacts.S3ModelArtifacts
    return Param(name=name, source=source)


class Pipeline:

    def __init__(
            self, name, root_dir, image_uri, bucket, role, region,
            pipeline_session
    ):
        self.name = name
        self.root_path = root_dir
        self.image_uri = image_uri
        self.bucket = bucket
        self.role = role
        self.region = region
        self.pipeline_session = pipeline_session

        current_date = datetime.now(pytz.utc).isoformat()[0:-6]
        self.base_s3_path = (
            f's3://{self.bucket}/pipelines/{self.name}/{current_date}'
        )
        self.steps = []

    def execute(self):
        sagemaker_pipeline = SageMakerPipeline(
            name=self.name,
            steps=self.steps,
        )

        sagemaker_pipeline.upsert(role_arn=self.role)
        execution = sagemaker_pipeline.start()
        print(execution)

    def add_processing_step(self, step: ProcessingFacadeStep):
        ProcessingStepParser(
            root_dir=self.root_path,
            image_uri=self.image_uri,
            role=self.role,
            pipeline_session=self.pipeline_session
        ).parse(step)
        self.steps.append(step.parsed_step)

    def add_training_step(self, step: TrainingFacadeStep):
        TrainingStepParser(
            base_s3_path=self.base_s3_path,
            image_uri=self.image_uri,
            role=self.role,
            region=self.region,
            pipeline_session=self.pipeline_session
        ).parse(step)
        self.steps.append(step.parsed_step)
