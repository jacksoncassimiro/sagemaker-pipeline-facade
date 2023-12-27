from sagemaker_pipeline_facade import Param, FacadeStep


class RegisterFacadeStep(FacadeStep):
    def __init__(
            self, model: Param, steps: dict,
            group_name: str, approval_status: str,
            content_type: str,
            response_type: str
    ):
        super().__init__()
        self.model = model.source
        self.steps = steps
        self.group_name = group_name
        self.approval_status = approval_status
        self.content_type = content_type
        self.response_type = response_type

    def build_metrics(self) -> dict:
        pass
