import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import os
import shutil
import time
"""
python grounded_sam2_tracking_demo_with_continuous_id.py

"""
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()

def save_video_frames(video_path: str, frame_dir: str) -> None:
    """
    주어진 비디오 파일에서 모든 프레임을 추출하여 지정된 디렉토리에 저장하는 함수.

    각 프레임은 6자리 숫자로 된 이름 (000000.jpg, 000001.jpg, ...) 으로 JPG 파일로 저장됩니다.
    frame_dir 폴더가 없거나 내용물이 있으면, 폴더를 빈 상태로 초기화한 후 저장을 시작합니다.

    Args:
        video_path (str): 입력 비디오 파일 경로 (예: "./video/input.mp4").
        frame_dir (str): 프레임을 저장할 디렉토리 경로 (예: "./frames").

    Returns:
        None: 프레임을 저장하며 반환값은 없습니다.
    """
    # 디렉토리가 존재할 경우 내부 파일을 모두 삭제하여 초기화
    if os.path.exists(frame_dir):
        return
    os.makedirs(frame_dir)  # 빈 디렉토리 생성

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0  # 프레임 번호

    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        # 파일명 생성 (6자리 숫자 형식, 000000.jpg, 000001.jpg ...)
        frame_name = f"{frame_idx:06}.jpg"
        frame_path = os.path.join(frame_dir, frame_name)

        # 프레임을 JPG로 저장
        cv2.imwrite(frame_path, frame)

        # 다음 프레임으로
        frame_idx += 1

    # 비디오 캡처 객체 해제
    cap.release()


"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "ball . person . "

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_path = "./video/input.mp4"
video_dir = "video_frames"

# 'output_dir' is the directory to save the annotated frames
output_dir = "./outputs"
# 'output_video_path' is the path to save the final video
output_video_path = "./outputs/output.mp4"
# create the output directory
# rm
shutil.rmtree(output_dir, ignore_errors=True)
CommonUtils.creat_dirs(output_dir)
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
# scan all the JPEG frame names in this directory
save_video_frames(video_path, video_dir)

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=False, async_loading_frames=False)
step = 20 # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
start_time = time.time()

for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    # continue
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path)
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"] # .cpu().numpy()
    # print("results[0]",results[0])
    OBJECTS = results[0]["labels"]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 2:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor
    """

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
    else:
        raise NotImplementedError("SAM 2 video predictor only support mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    print("objects_count", objects_count)
    video_predictor.reset_state(inference_state)
    if len(mask_dict.labels) == 0:
        print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
        continue
    video_predictor.reset_state(inference_state)

    for object_id, object_info in mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
    
    video_segments = {}  # output the following {step} frames tracking masks
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
        frame_masks = MaskDictionaryModel()
        
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
            object_info.update_box()
            frame_masks.labels[out_obj_id] = object_info
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            frame_masks.mask_name = f"mask_{image_base_name}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        video_segments[out_frame_idx] = frame_masks
        sam2_masks = copy.deepcopy(frame_masks)

    print("video_segments:", len(video_segments))
    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data = frame_masks_info.to_dict()
        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        with open(json_data_path, "w") as f:
            json.dump(json_data, f)

# remove video_dir.
shutil.rmtree(video_dir)

end_time = time.time()
print("Total time:", end_time - start_time, "total frames:", len(frame_names))



"""
Step 6: Draw the results and save the video
"""
CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

create_video_from_images(result_dir, output_video_path, frame_rate=15)