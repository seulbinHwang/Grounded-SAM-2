import cv2
import torch
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import pickle
from typing import List


def save_ndarray_list(a_var: List[np.ndarray], file_path: str) -> None:
    """np.ndarray의 리스트를 파일로 저장하는 함수.

    Args:
        a_var (List[np.ndarray]): 저장할 numpy ndarray 리스트.
        file_path (str): 저장할 파일 경로.

    Raises:
        IOError: 파일 저장 중 문제가 발생한 경우.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(a_var, f)
    except IOError as e:
        raise IOError(f"파일을 저장하는 중 오류가 발생했습니다: {e}")


def load_ndarray_list(file_path: str) -> List[np.ndarray]:
    """파일에서 np.ndarray 리스트를 불러오는 함수.

    Args:
        file_path (str): 불러올 파일 경로.

    Returns:
        List[np.ndarray]: 파일에서 불러온 numpy ndarray 리스트.

    Raises:
        IOError: 파일 불러오기 중 문제가 발생한 경우.
        ValueError: 파일의 내용이 예상한 데이터 형식이 아닌 경우.
    """
    try:
        with open(file_path, 'rb') as f:
            a_var = pickle.load(f)
            if not isinstance(a_var, list) or not all(
                    isinstance(i, np.ndarray) for i in a_var):
                raise ValueError("불러온 데이터가 올바른 형식이 아닙니다.")
            return a_var
    except IOError as e:
        raise IOError(f"파일을 불러오는 중 오류가 발생했습니다: {e}")


# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id).to(device)

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
#text = "car. tire."
text = "field."
img_parents = ["left_frames", "right_frames"]
pngs_number = None
left_batch_masks: List[np.ndarray] = []
right_batch_masks = []
for parent_idx, img_parent in enumerate(img_parents):
    all_pngs = os.listdir(img_parent)
    img_parent_result = f"{parent_idx}_result"
    if not os.path.exists(img_parent_result):
        os.makedirs(img_parent_result)
    if pngs_number is None:
        pngs_number = len(all_pngs)
    else:
        assert pngs_number == len(
            all_pngs
        ), "The number of png files in the directories should be the same."
    for img_idx, png_img in enumerate(all_pngs):
        img_path = f"{img_parent}/{png_img}"
        image = Image.open(img_path)
        sam2_predictor.set_image(np.array(image.convert("RGB")))

        inputs = processor(images=image, text=text,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]])
        """
        Results is a list of dict with the following structure:
        [
            {
                'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
                'labels': ['car', 'tire', 'tire', 'tire'], 
                'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                                [1392.4701,  554.4064, 1628.6133,  777.5872],
                                [ 436.1182,  621.8940,  676.5255,  851.6897],
                                [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
            }
        ]
        """

        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if parent_idx == 0:
            left_batch_masks.append(masks)
        else:
            right_batch_masks.append(masks)
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]
        """
        Visualize image with supervision useful API
        """
        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids)
        """
        Note that if you want to use default color map,
        you can set color=ColorPalette.DEFAULT
        """
        box_annotator = sv.BoxAnnotator(
            color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(),
                                                 detections=detections)

        label_annotator = sv.LabelAnnotator(
            color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame,
                                                   detections=detections,
                                                   labels=labels)
        cv2.imwrite(
            f"{img_parent_result}/groundingdino_annotated_image_{img_idx}.jpg",
            annotated_frame)

        mask_annotator = sv.MaskAnnotator(
            color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame,
                                                  detections=detections)
        cv2.imwrite(
            f"{img_parent_result}/grounded_sam2_annotated_image_with_mask_{img_idx}.jpg",
            annotated_frame)

save_ndarray_list(left_batch_masks, "left_batch_masks.pkl")
save_ndarray_list(right_batch_masks, "right_batch_masks.pkl")
