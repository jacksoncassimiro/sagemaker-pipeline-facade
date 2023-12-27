from sagemaker import ModelMetrics, MetricsSource

from sagemaker_pipeline_facade.register_step import (
    RegisterFacadeStep
)


class RegisterModelFacadeStep(RegisterFacadeStep):

    def build_metrics(self) -> dict:
        evaluation_step = self.steps['evaluation_step'].parsed_step
        evaluation_s3_path = (
            evaluation_step.arguments["ProcessingOutputConfig"]
            ["Outputs"][0]["S3Output"]["S3Uri"]
        )
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f'{evaluation_s3_path}/evaluation.json',
                content_type="application/json",
            )
        )
        return {'model_metrics': model_metrics}