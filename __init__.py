from .node import *

NODE_CLASS_MAPPINGS = {
    'MaskAreaComparisonSegment': MaskAreaComparisonSegment,
    'FillMaskedArea': FillMaskedArea,
    'OCRAndMask': DetectAndMask,
}

__all__ = ['NODE_CLASS_MAPPINGS']

