from datetime import datetime

import pytz
from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline

from sagemaker_pipeline_facade.processing_step import ProcessingFacadeStep
from sagemaker_pipeline_facade.processing_step_parser import \
    ProcessingParser
from sagemaker_pipeline_facade.training_step import TrainingFacadeStep


class Pipeline:

    def __init__(
            self, name, root_dir, bucket, role, region, pipeline_session
    ):
        self.name = name
        self.root_path = root_dir
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
        ProcessingParser(
            root_dir=self.root_path,
            role=self.role,
            pipeline_session=self.pipeline_session
        ).parse(step)
        self.steps.append(step.parsed_step)

    def add_training_step(self, step: TrainingFacadeStep):
        step.parse(
            base_s3_path=self.base_s3_path,
            role=self.role,
            region=self.region,
            pipeline_session=self.pipeline_session
        )
        self.steps.append(step.parsed_step)
