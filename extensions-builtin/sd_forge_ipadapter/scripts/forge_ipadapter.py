import math

import cv2
import numpy as np
import torch

from pathlib import Path

from lib_ipadapter.IPAdapterPlus import IPAdapterApply, InsightFaceLoader

from modules_forge.supported_preprocessor import PreprocessorClipVision, Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, add_supported_control_model
from modules_forge.utils import numpy_to_pytorch, resize_image_with_pad
from modules_forge.supported_controlnet import ControlModelPatcher


class PreprocessorClipVisionForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.tags = ["IP-Adapter"]
        self.model_filename_filters = ["IP-Adapter", "IP_Adapter"]
        self.sorting_priority = 20

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            unfold_batch=False,
        )
        return cond


class PreprocessorClipVisionWithInsightFaceForIPAdapter(PreprocessorClipVisionForIPAdapter):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.cached_insightface = None

    def load_insightface(self):
        if self.cached_insightface is None:
            self.cached_insightface = InsightFaceLoader.load_insight_face()
        return self.cached_insightface

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            insightface=self.load_insightface(),
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            unfold_batch=False,
        )
        return cond


class PreprocessorInsightFaceForInstantID(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ["Instant-ID"]
        self.model_filename_filters = ["Instant-ID", "Instant_ID"]
        self.sorting_priority = 20
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.show_control_mode = False
        self.sorting_priority = 10
        self.cached_insightface = None
        self.latest_face_embedding = None

    @staticmethod
    def draw_keypoints(img, keypoints, color_list=((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255))):
        stick_width = 4
        limb_sequence = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        keypoints = np.array(keypoints)

        height, width, _ = img.shape
        output_image = np.zeros((height, width, 3), dtype=np.uint8)

        for index in limb_sequence:
            color = color_list[index[0]]
            x_coords = keypoints[index][:, 0]
            y_coords = keypoints[index][:, 1]
            length = ((x_coords[0] - x_coords[1]) ** 2 + (y_coords[0] - y_coords[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y_coords[0] - y_coords[1], x_coords[0] - x_coords[1]))
            polygon = cv2.ellipse2Poly(
                (int(np.mean(x_coords)), int(np.mean(y_coords))),
                (int(length / 2), stick_width),
                int(angle),
                0,
                360,
                1,
            )
            output_image = cv2.fillConvexPoly(output_image.copy(), polygon, color)

        output_image = (output_image * 0.6).astype(np.uint8)

        for keypoint_index, keypoint in enumerate(keypoints):
            color = color_list[keypoint_index]
            x_coord, y_coord = keypoint
            output_image = cv2.circle(output_image.copy(), (int(x_coord), int(y_coord)), 10, color, -1)

        return output_image

    def load_insightface(self):
        if self.cached_insightface is None:
            self.cached_insightface = InsightFaceLoader.load_insight_face(name="antelopev2")
        return self.cached_insightface

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        insightface = self.load_insightface()
        resized_image, remove_pad = resize_image_with_pad(input_image, resolution)
        face_info = insightface.get(resized_image)

        if not face_info:
            raise Exception("InsightFace: No face found in image.")

        if len(face_info) > 1:
            print("InsightFace: More than one face is detected in the image. Only the biggest one will be used.")

        primary_face = max(
            face_info,
            key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]),
        )

        self.latest_face_embedding = torch.from_numpy(primary_face['embedding']).float().unsqueeze(0)

        control_image = remove_pad(self.draw_keypoints(resized_image, primary_face['kps']))

        return {
            "image": numpy_to_pytorch(control_image).movedim(-1, 1),
            "y": self.latest_face_embedding,
        }


add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(name="CLIP-ViT-H (IPAdapter)", url="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", filename="CLIP-ViT-H-14.safetensors"))

add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(name="CLIP-ViT-bigG (IPAdapter)", url="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors", filename="CLIP-ViT-bigG.safetensors"))

add_supported_preprocessor(PreprocessorClipVisionWithInsightFaceForIPAdapter(name="InsightFace+CLIP-H (IPAdapter)", url="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", filename="CLIP-ViT-H-14.safetensors"))

add_supported_preprocessor(PreprocessorInsightFaceForInstantID(name="InsightFace (InstantID)"))


class IPAdapterPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        model = state_dict

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        if "ip_adapter" not in model.keys() or len(model["ip_adapter"]) == 0:
            return None

        o = IPAdapterPatcher(model)

        model_filename = Path(ckpt_path).name.lower()
        if "v2" in model_filename:
            o.faceid_v2 = True
            o.weight_v2 = True

        return o

    def __init__(self, state_dict):
        super().__init__()
        self.ip_adapter = state_dict
        self.faceid_v2 = False
        self.weight_v2 = False
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = IPAdapterApply.apply_ipadapter(
            ipadapter=self.ip_adapter,
            model=unet,
            weight=self.strength,
            start_at=self.start_percent,
            end_at=self.end_percent,
            faceid_v2=self.faceid_v2,
            weight_v2=self.weight_v2,
            attn_mask=mask.squeeze(1) if mask is not None else None,
            **cond,
        )

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(IPAdapterPatcher)
