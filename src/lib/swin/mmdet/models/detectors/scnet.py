from ..builder import DETECTORS
from .cascade_rcnn import CascadeRCNN


@DETECTORS._register()
class SCNet(CascadeRCNN):
    """Implementation of `SCNet <https://arxiv.org/abs/2012.10150>`_"""

    def __init__(self, **kwargs):
        super(SCNet, self).__init__(**kwargs)
