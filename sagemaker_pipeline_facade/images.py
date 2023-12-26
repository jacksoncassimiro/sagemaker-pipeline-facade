from dataclasses import dataclass
from enum import Enum
import sagemaker


@dataclass
class ImageArgs:
    version: str
    py_version: str


class ImageLoader(Enum):
    xgboost: ImageArgs = ImageArgs(
        version='1.0-1', py_version='py3'
    )

    def load(self, region):
        return sagemaker.image_uris.retrieve(
            region=region,
            framework=self.name,
            version=self.value.version,
            py_version=self.value.py_version,
        )