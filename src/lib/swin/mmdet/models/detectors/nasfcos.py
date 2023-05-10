from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS._register()
class NASFCOS(SingleStageDetector):
    """NAS-FCOS: Fast Neural Architecture Search for Object Detection.

    https://arxiv.org/abs/1906.0442
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(NASFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained)