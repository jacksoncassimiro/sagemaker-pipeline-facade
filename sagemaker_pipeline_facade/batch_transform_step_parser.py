import sagemaker
from sagemaker import Model
from sagemaker.inputs import TransformInput
from sagemaker.transformer import Transformer
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import TransformStep

from sagemaker_pipeline_facade.batch_transform_step import (
    BatchTransformFacadeStep
)
from sagemaker_pipeline_facade import (
    DEFAULT_INSTANCE_TYPE, DEFAULT_INSTANCE_COUNT
)


class BatchTransformStepParser:
    def __init__(self, base_s3_path, image_uri, role, pipeline_session):
        self.base_s3_path = base_s3_path
        self.image_uri = image_uri
        self.role = role
        self.pipeline_session = pipeline_session

    def parse(self, step: BatchTransformFacadeStep):
        model = Model(
            image_uri=self.image_uri,
            model_data=step.model_data.source,
            sagemaker_session=self.pipeline_session,
            role=self.role,
        )

        model_step = ModelStep(
            name=step.name(),
            step_args=model.create(instance_type=DEFAULT_INSTANCE_TYPE),
        )

        transformer = Transformer(
            model_name=model_step.properties.ModelName,
            instance_type=DEFAULT_INSTANCE_TYPE,
            instance_count=DEFAULT_INSTANCE_COUNT,
            output_path=f'{self.base_s3_path}/transform/output',
        )

        data_uri = step.batch_data.source

        if not data_uri.startswith('s3://'):
            data_uri = sagemaker.s3.S3Uploader.upload(
                local_path=step.batch_data.source,
                desired_s3_uri=f'{self.base_s3_path}/transform/data',
            )

        transform_step = TransformStep(
            name=f'{step.name()}-Transform',
            transformer=transformer,
            inputs=TransformInput(
                data=data_uri,
            ),
        )

        step.model = model
        step.parsed_model_step = model_step
        step.parsed_transform_step = transform_step

        return model_step, transform_step
