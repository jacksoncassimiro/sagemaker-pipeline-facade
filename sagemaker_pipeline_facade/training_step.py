from dataclasses import dataclass
from enum import Enum

import sagemaker
from sagemaker import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import TrainingStep

from sagemaker_pipeline_facade import Param, FacadeStep


@dataclass
class ImageArgs:
    version: str
    py_version: str
    instance_type: str


@dataclass
class ContainerArgs:
    instance_count: int


@dataclass
class ModelArgs:
    name: str
    hyper_params: dict


class ImageArgsPerModel(Enum):
    xgboost: ImageArgs = ImageArgs(
        version='1.0-1', py_version='py3', instance_type='ml.m5.xlarge'
    )


DEFAULT_CONTAINER_ARGS = ContainerArgs(instance_count=1)


class TrainingFacadeStep(FacadeStep):

    def __init__(
            self,
            inputs: list[Param],
            model_args: ModelArgs,
            image_args: ImageArgs = None,
            container_args: ContainerArgs = None
    ):
        self.parsed_step = None
        self.models_path = None
        self.role = None
        self.region = None
        self.pipeline_session = None

        self.inputs = inputs
        self.model_args = model_args

        if not image_args:
            self.image_args = ImageArgsPerModel[model_args.name].value

        if not container_args:
            self.container_args = DEFAULT_CONTAINER_ARGS

    def parse(self, base_s3_path, role, region, pipeline_session):
        self.models_path = f'{base_s3_path}/models'
        self.role = role
        self.region = region
        self.pipeline_session = pipeline_session

        image_uri = sagemaker.image_uris.retrieve(**self.get_image_args())

        estimator = Estimator(
            image_uri=image_uri,
            instance_type=self.image_args.instance_type,
            instance_count=self.container_args.instance_count,
            output_path=self.models_path,
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

        estimator.set_hyperparameters(**self.get_hyper_params())

        train_args = estimator.fit(
            inputs={
                param.name: TrainingInput(
                    s3_data=param.source,
                    content_type=param.content_type,
                )
                for param in self.inputs
            }
        )

        self.parsed_step = TrainingStep(
            name=self.name(),
            step_args=train_args,
        )

        return self.parsed_step

    def get_image_args(self):
        return {
            'region': self.region,
            'framework': self.model_args.name,
            'version': self.image_args.version,
            'py_version': self.image_args.py_version,
            'instance_type': self.image_args.instance_type,
        }

    def get_hyper_params(self):
        return self.model_args.hyper_params
