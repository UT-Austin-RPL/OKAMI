import numpy as np
import cv2
import os
import time

import mujoco
import mujoco.viewer

from okami.oar.algos.gr1 import GR1URDFModel
from okami.oar.utils.frame_transformation import GR1Transform
from okami.oar.utils.urdf_utils import obs_to_urdf
from okami.oar.utils.o3d_utils import O3DPointCloud, visualize_o3d_point_cloud
import open3d as o3d

import torch
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 

# diffusers
import requests
from io import BytesIO

from huggingface_hub import hf_hub_download

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sam_checkpoint_path = os.path.join(root_path, './third_party/sam_checkpoints/sam_vit_h_4b8939.pth' )
print("root_path: ", root_path)
print("sam_checkpoint_path: ", sam_checkpoint_path)

class ObjectLocalizer:
    def __init__(self, transform: GR1Transform, gsam_wrapper):
        self.transform = transform
        self.wrapper = gsam_wrapper

    def localize(self, obj_name, obs):
        if self.wrapper is None:
            return self.localize_by_state(obj_name, obs)
        else:
            return self.localize_by_vision(obj_name, obs)

    def localize_by_state(self, obj_name, obs):
        '''
        Return the position of the center of the object with respect to the head of the robot
        '''
        print("localize by state")  

        object_pos_world = obs[obj_name + '_pos']
        # print("object pos world: ", object_pos_world)
        object_pos_head = self.transform.apply_transform_to_point(object_pos_world, 'world', 'head')
        return object_pos_head
    
    def localize_by_vision(self, obj_name, obs):
        '''
        Return the position of the center of the object with respect to the head of the robot
        '''
        print("localize by vision")

        assert('robot0_robotview_image' in obs)
        assert('robot0_robotview_depth' in obs)
        rgb_img = obs['robot0_robotview_image']
        depth_img = obs['robot0_robotview_depth']

        rgb_img = cv2.flip(rgb_img, 0)
        depth_img = cv2.flip(depth_img, 0)
        
        # Communicate with notebook and run G-SAM to localize the object
        final_mask_image = self.wrapper.segment(rgb_img, [obj_name])
        overlay_image = overlay_xmem_mask_on_image(rgb_img, np.array(final_mask_image), use_white_bg=True, rgb_alpha=0.3)
        # save image
        cv2.imwrite("overlay_image.png", overlay_image)

        # decode the depth image
        extent = 10.60981561453384 
        near = 0.0010000000474974513 * extent 
        far = 50.0 * extent
        depth_img = (near / (1 - depth_img * (1 - near / far))).copy()

        binary_mask = np.array(final_mask_image) > 0
        depth_img = depth_img.reshape((depth_img.shape[0], depth_img.shape[1]))
        masked_depth = (depth_img * binary_mask).copy()

        with open('pcd_info.npz', 'wb') as f:
            np.savez(f, depth_image=masked_depth, 
                        binary_mask=binary_mask,
                        camera_intrinsics=self.transform.camera_intrinsics, 
                        camera_extrinsics=self.transform.camera_extrinsics,
                        img=rgb_img)

        rgbd_pc = O3DPointCloud(max_points=50000)
        rgbd_pc.create_from_rgbd(rgb_img, masked_depth * 1000, self.transform.camera_intrinsics, depth_trunc=3.0)
        rgbd_pc.transform(self.transform.camera_extrinsics) # transform from camera frame to head frame
        rgbd_pc.preprocess()

        pcd_points = rgbd_pc.get_points()
        pcd_centers = np.mean(pcd_points, axis=0)

        # gt = self.localize_by_state(obj_name, obs)
        # print("\n pcd_centers: ", pcd_centers, "object_gt_pos_in_head=", gt)

        return pcd_centers


class GroundedSamWrapper:
    def load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cuda'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file) 
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cuda')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model   
    
    def __init__(self):
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        self.groundingdino_model = self.load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        device = 'cuda'
        sam_checkpoint = sam_checkpoint_path
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def transform(self, image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(image).convert("RGB")
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed

    def combine_masks(self, masks):
        final_mask = np.zeros((masks.shape[2], masks.shape[3]), dtype=np.uint8)
        masks = masks.cpu().detach().numpy()
        final_mask = final_mask + masks[0][0]

        for i in range(1, masks.shape[0]):
            final_mask += (masks[i][0] * (i + 1)).astype(np.uint8)
        
        #for visualizing the mask
        #final_mask = final_mask * (255 // np.amax(final_mask))

        return final_mask

    def segment(self, image_np, prompts, box_threshold=0.3, text_threshold=0.25, filter_threshold=200):
        image_source, image = self.transform(image_np)

        prompt_text = ""
        for prompt in prompts:
            prompt_text += (prompt + ".")

        boxes, logits, phrases = predict(
            model=self.groundingdino_model, 
            image=image, 
            caption=prompt_text, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold
        )

        if boxes.shape[0] == 0:
            print("no boxes found!")
            return np.array([])

           
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB

        img_src = Image.fromarray(image_source)
        # img_src.save("img_src.png")

        img_annotated = Image.fromarray(annotated_frame)
        # img_annotated.save("annotated.png")

        # set image
        self.sam_predictor.set_image(image_source)

        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        device = "cuda"
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )

        intermediate_final_mask = self.combine_masks(masks)

        filter_indices = []
        for i in range(1, intermediate_final_mask.max() + 1):
            if np.sum(intermediate_final_mask == i) < filter_threshold:
                filter_indices.append(i)
        
        final_mask = np.zeros_like(intermediate_final_mask)
        count = 0
        for i in range(1, intermediate_final_mask.max() + 1):
            if i not in filter_indices:
                final_mask[intermediate_final_mask == i] = count + 1
                count += 1

        mask_image_pil = Image.fromarray(final_mask) # .convert("RGBA")
        mask_image_pil.putpalette(get_palette())
        # mask_image_pil.save("final_mask.png")
        return mask_image_pil

def get_palette(palette="davis"):
    davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'
    youtube_palette = b'\x00\x00\x00\xec_g\xf9\x91W\xfa\xc8c\x99\xc7\x94b\xb3\xb2f\x99\xcc\xc5\x94\xc5\xabyg\xff\xff\xffes~\x0b\x0b\x0b\x0c\x0c\x0c\r\r\r\x0e\x0e\x0e\x0f\x0f\x0f'
    if palette == "davis":
        return davis_palette
    elif palette == "youtube":
        return youtube_palette

def overlay_xmem_mask_on_image(rgb_img, mask, use_white_bg=False, rgb_alpha=0.7):
    """

    Args:
        rgb_img (np.ndarray):rgb images
        mask (np.ndarray)): binary mask
        use_white_bg (bool, optional): Use white backgrounds to visualize overlap. Note that we assume mask ids 0 as the backgrounds. Otherwise the visualization might be screws up. . Defaults to False.

    Returns:
        np.ndarray: overlay image of rgb_img and mask
    """
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    if use_white_bg:
        colored_mask[mask == 0] = [255, 255, 255]
    overlay_img = cv2.addWeighted(rgb_img, rgb_alpha, colored_mask, 1-rgb_alpha, 0)

    return overlay_img