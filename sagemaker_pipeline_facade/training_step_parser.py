from sagemaker import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import TrainingStep

from sagemaker_pipeline_facade import (
    DEFAULT_INSTANCE_TYPE, DEFAULT_INSTANCE_COUNT
)
from sagemaker_pipeline_facade.training_step import TrainingFacadeStep


class TrainingStepParser:
    def __init__(
            self, base_s3_path, image_uri, role, region, pipeline_session
    ):
        self.base_s3_path = base_s3_path
        self.image_uri = image_uri
        self.role = role
        self.region = region
        self.pipeline_session = pipeline_session

    def parse(self, step: TrainingFacadeStep):
        estimator = self.get_estimator()
        estimator.set_hyperparameters(**step.hyper_params)

        train_args = estimator.fit(
            inputs={
                param.name: TrainingInput(
                    s3_data=param.source,
                    content_type=param.content_type,
                )
                for param in step.inputs
            }
        )

        step.parsed_step = TrainingStep(
            name=step.name(),
            step_args=train_args,
        )

        return step.parsed_step

    def get_estimator(self):
        return Estimator(
            image_uri=self.image_uri,
            instance_type=DEFAULT_INSTANCE_TYPE,
            instance_count=DEFAULT_INSTANCE_COUNT,
            output_path=f'{self.base_s3_path}/models',
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )
