from sagemaker_pipeline_facade.register_step import RegisterFacadeStep
from sagemaker.workflow.model_step import ModelStep


class RegisterStepParser:

    def parse(self, step: RegisterFacadeStep):
        model = step.model
        metrics = step.build_metrics()

        args = model.register(
            content_types=[step.content_type],
            response_types=[step.response_type],
            model_package_group_name=step.group_name,
            approval_status=step.approval_status,
            **metrics
        )

        step.parsed_step = ModelStep(name=step.name(), step_args=args)
        return step.parsed_step
