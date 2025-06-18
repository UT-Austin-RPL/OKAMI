import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import init_path
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from okami.plan_generation.utils.misc_utils import get_annotation_path, get_first_frame_annotation, VideoWriter, overlay_xmem_mask_on_image, add_palette_on_mask, get_video_seq_from_annotation, get_palette

class CutieTracker:
    """
    This class provides a wrapper for the Cutie tracking model with an interface similar to XMemTracker.
    """
    def __init__(self, device, half_mode=False) -> None:
        """
        device: model device
        cutie_checkpoint: checkpoint of Cutie model
        """
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.device = device
        self.half_mode = half_mode

        if half_mode:
            self.cutie.half()

        self.palette = get_palette()
        # self.im_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frame: numpy array (H, W, 3)
        first_frame_annotation: numpy array (H, W), annotation of the first frame

        Output:
        mask: numpy array (H, W)
        """
        if first_frame_annotation is not None:
            mask = torch.from_numpy(first_frame_annotation).to(self.device)
            objects = np.unique(first_frame_annotation)[1:].tolist()  # Exclude background labeled as "0"
        else:
            mask = None
            objects = []

        frame_tensor = to_tensor(Image.fromarray(frame)).to(self.device).float()

        if self.half_mode and frame_tensor.dtype != torch.float16:
            frame_tensor = frame_tensor.half()
            if mask is not None:
                mask = mask.half()

        if mask is not None:
            output_prob = self.processor.step(frame_tensor, mask, objects=objects)
        else:
            output_prob = self.processor.step(frame_tensor)

        mask_np = self.processor.output_prob_to_mask(output_prob).cpu().numpy().astype(np.uint8)

        return mask_np

    def track_video(self, video_frames, initial_mask):
        """
        Track a series of images in a single function.
        """
        masks = []
        for (i, frame) in enumerate(tqdm(video_frames)):
            if i == 0:
                mask = self.track(frame, initial_mask)
            else:
                mask = self.track(frame)
            masks.append(mask)
        return masks

    @torch.no_grad()
    def clear_memory(self):
        self.processor.clear_memory()
        torch.cuda.empty_cache()

    def save_colored_masks(self, masks, output_dir):
        """
        Save the sequence of masks to the output directory with colors.
        Args:
            masks (list of np.array): List of H x W logits
            output_dir (str): Output directory path
        """
        for i, mask in enumerate(masks):
            colored_mask = Image.fromarray(mask)
            colored_mask.putpalette(self.palette)
            colored_mask.save(os.path.join(output_dir, f"{i:07d}.png"))
