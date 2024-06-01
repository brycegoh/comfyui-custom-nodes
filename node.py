import torch
from PIL import Image
import numpy as np
import cv2
class MaskAreaComparisonSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "image": ("IMAGE", {}),
                "if_mask_smaller": ("IMAGE", {}),
                "if_mask_larger": ("IMAGE", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    CATEGORY = "segment_analysis"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)

    def main(self, mask, image, if_mask_smaller, if_mask_larger, threshold):
        # Convert image to a numpy array if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        elif not isinstance(image, np.ndarray):
            image = np.asarray(image)

        # Convert mask to a numpy array if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        elif not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)

        # Calculate the total area of the image
        total_area = image.shape[1] * image.shape[2]

        # Calculate the mask's area by counting non-zero pixels
        mask_area = np.count_nonzero(mask)

        # Calculate % of the image's total area
        threshold_area = threshold * total_area

        # Compare mask area with threshold area
        if mask_area < threshold_area:
            return (if_mask_smaller,)
        else:
            return (if_mask_larger,)

class FillMaskedArea:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill": (["black"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "fill"

    def fill(self, image: torch.Tensor, mask: torch.Tensor, fill: str):
        image = image.detach().clone()
        alpha = mask.expand(1, *mask.shape[-2:]).floor()
        if fill == "black":
            for i in range(3):  # Apply black to each channel
                image[:, :, :, i] *= (1 - alpha.squeeze(1))
        return (image,)



class DetectAndMask:
  @classmethod
  def INPUT_TYPES(s):
    return {
        "required": {
            "images": ("IMAGE",),
            "text": ("STRING", {"multiline": False, "dynamicPrompts": False})
        }
    }

  RETURN_TYPES = ("MASK",)
  CATEGORY = "mask"
  FUNCTION = "detect_and_mask"

  def detect_and_mask(self, images: torch.Tensor, text: str):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    for (batch_number, image) in enumerate(images):
      i = 255. * image.cpu().numpy()
      img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
      np_img = np.array(img)
      ocr_results = ocr.ocr(np_img, cls=True)

      mask = np.zeros(np_img.shape[:2], dtype=np.uint8)

      for line in ocr_results:
        for box, (detected_text, confidence) in line:
          if text.lower() in detected_text.lower():
            box = np.array(box).astype(np.int32)
            horizontal_padding = 80
            vertical_padding = 10
            # Calculate padding
            x_min = max(np.min(box[:, 0]) - horizontal_padding, 0)
            x_max = min(np.max(box[:, 0]) + horizontal_padding, mask.shape[1])
            y_min = max(np.min(box[:, 1]) - vertical_padding, 0)
            y_max = min(np.max(box[:, 1]) + vertical_padding, mask.shape[0])
            
            # Fill the padded area on the mask
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, thickness=-1)
      mask_tensor = torch.from_numpy(mask).unsqueeze(0)
      return (mask_tensor,)
