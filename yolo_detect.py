import os
import sys
import argparse
import glob
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser(
    description="YOLOv8 Metal Surface Defect Detection"
)

parser.add_argument(
    "--model",
    help="Path to YOLO model file",
    required=True
)

parser.add_argument(
    "--source",
    help=(
        "Image file, image folder, video file, "
        "USB camera such as usb0"
    ),
    required=True
)

parser.add_argument(
    "--thresh",
    type=float,
    help="Minimum confidence threshold",
    default=0.5
)

parser.add_argument(
    "--resolution",
    help='Optional display resolution such as "1280x720"',
    default=None
)

parser.add_argument(
    "--record",
    action="store_true",
    help="Record video/webcam output"
)

args = parser.parse_args()


# ============================================================
# USER INPUTS
# ============================================================

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(model_path):

    print("ERROR: Model file was not found.")
    print("Model path:", model_path)

    sys.exit(0)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING YOLO11s MODEL")
print("=" * 70)

model = YOLO(
    model_path,
    task="detect"
)

labels = model.names

print("Model loaded successfully.")

print("\nDefect Classes:")

for class_id, class_name in labels.items():

    print(
        f"  {class_id}: {class_name}"
    )


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

img_ext_list = [
    ".jpg", ".JPG",
    ".jpeg", ".JPEG",
    ".png", ".PNG",
    ".bmp", ".BMP",
    ".webp", ".WEBP"
]

vid_ext_list = [
    ".avi", ".AVI",
    ".mov", ".MOV",
    ".mp4", ".MP4",
    ".mkv", ".MKV",
    ".wmv", ".WMV"
]


# ============================================================
# DETERMINE SOURCE TYPE
# ============================================================

if os.path.isdir(img_source):

    source_type = "folder"

elif os.path.isfile(img_source):

    _, ext = os.path.splitext(img_source)

    if ext in img_ext_list:

        source_type = "image"

    elif ext in vid_ext_list:

        source_type = "video"

    else:

        print(
            f"ERROR: File extension {ext} is not supported."
        )

        sys.exit(0)

elif "usb" in img_source.lower():

    source_type = "usb"

    try:

        usb_idx = int(
            img_source[3:]
        )

    except ValueError:

        print(
            "ERROR: USB camera must be specified "
            "like usb0 or usb1."
        )

        sys.exit(0)

else:

    print(
        f"ERROR: Input '{img_source}' is invalid."
    )

    sys.exit(0)


# ============================================================
# OPTIONAL RESOLUTION
# ============================================================

user_res_enabled = False

if user_res:

    try:

        resW, resH = map(
            int,
            user_res.lower().split("x")
        )

        if resW <= 0 or resH <= 0:
            raise ValueError

        user_res_enabled = True

    except ValueError:

        print(
            'ERROR: Resolution must be like "1280x720".'
        )

        sys.exit(0)


# ============================================================
# RECORDING
# ============================================================

if record:

    if source_type not in ["video", "usb"]:

        print(
            "ERROR: Recording is available only "
            "for video and USB camera."
        )

        sys.exit(0)

    if not user_res_enabled:

        print(
            "ERROR: Specify --resolution when recording."
        )

        sys.exit(0)


if record:

    record_name = "demo1.avi"

    record_fps = 30

    recorder = cv2.VideoWriter(
        record_name,
        cv2.VideoWriter_fourcc(*"MJPG"),
        record_fps,
        (resW, resH)
    )


# ============================================================
# LOAD SOURCE
# ============================================================

if source_type == "image":

    imgs_list = [
        img_source
    ]


elif source_type == "folder":

    imgs_list = []

    filelist = glob.glob(
        os.path.join(
            img_source,
            "*"
        )
    )

    for file in filelist:

        _, file_ext = os.path.splitext(file)

        if file_ext in img_ext_list:

            imgs_list.append(file)

    imgs_list.sort()

    if not imgs_list:

        print(
            "ERROR: No images found."
        )

        sys.exit(0)


elif source_type in ["video", "usb"]:

    if source_type == "video":

        cap_arg = img_source

    else:

        cap_arg = usb_idx

    cap = cv2.VideoCapture(
        cap_arg
    )

    if not cap.isOpened():

        print(
            "ERROR: Unable to open source."
        )

        sys.exit(0)

    if user_res_enabled:

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            resW
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            resH
        )


# ============================================================
# BOUNDING BOX COLORS
# ============================================================

bbox_colors = [
    (164, 120, 87),
    (68, 148, 228),
    (93, 97, 209),
    (178, 182, 133),
    (88, 159, 106),
    (96, 202, 231),
    (159, 124, 168),
    (169, 162, 241),
    (98, 118, 150),
    (172, 176, 184)
]


# ============================================================
# DISPLAY WINDOW
# ============================================================

WINDOW_NAME = (
    "YOLOv8 Metal Defect Detection"
)

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


# ============================================================
# AUTOMATIC SCREEN SIZE
# ============================================================

def get_screen_size():

    try:

        import tkinter as tk

        root = tk.Tk()

        root.withdraw()

        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        root.destroy()

        return width, height

    except Exception:

        return 1600, 900


SCREEN_WIDTH, SCREEN_HEIGHT = (
    get_screen_size()
)

MAX_DISPLAY_WIDTH = int(
    SCREEN_WIDTH * 0.90
)

MAX_DISPLAY_HEIGHT = int(
    SCREEN_HEIGHT * 0.80
)


# ============================================================
# FIT IMAGE TO SCREEN
# ============================================================

def fit_to_screen(
    image,
    max_width,
    max_height
):

    height, width = image.shape[:2]

    scale_width = (
        max_width / width
    )

    scale_height = (
        max_height / height
    )

    scale = min(
        scale_width,
        scale_height
    )

    # Never enlarge small images
    scale = min(
        scale,
        1.0
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    if (
        new_width == width
        and
        new_height == height
    ):

        return image

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# FPS
# ============================================================

avg_frame_rate = 0

frame_rate_buffer = []

fps_avg_len = 200

img_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    t_start = time.perf_counter()


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    if source_type in [
        "image",
        "folder"
    ]:

        if img_count >= len(imgs_list):

            print(
                "\nAll images have been processed."
            )

            break

        img_filename = (
            imgs_list[img_count]
        )

        frame = cv2.imread(
            img_filename
        )

        img_count += 1

        if frame is None:

            print(
                "WARNING: Unable to read:",
                img_filename
            )

            continue


    # ========================================================
    # VIDEO / USB
    # ========================================================

    elif source_type in [
        "video",
        "usb"
    ]:

        ret, frame = cap.read()

        if not ret or frame is None:

            print(
                "\nUnable to read frame."
            )

            break


    # ========================================================
    # ORIGINAL FRAME
    # ========================================================

    original_frame = frame.copy()


    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    results = model(
        original_frame,
        verbose=False
    )


    detections = results[0].boxes


    # ========================================================
    # DETECTION COUNTER
    # ========================================================

    object_count = 0

    detected_types = []

    detected_confidences = []


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for i in range(
        len(detections)
    ):

        conf = float(
            detections[i].conf.item()
        )


        # Ignore detections below threshold
        if conf < min_thresh:

            continue


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        xyxy = (
            detections[i]
            .xyxy
            .cpu()
            .numpy()
            .squeeze()
        )

        xmin, ymin, xmax, ymax = (
            xyxy.astype(int)
        )


        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        classidx = int(
            detections[i].cls.item()
        )

        classname = labels[
            classidx
        ]


        # ----------------------------------------------------
        # Store detection information
        # ----------------------------------------------------

        detected_types.append(
            classname
        )

        detected_confidences.append(
            conf
        )

        object_count += 1


        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        color = bbox_colors[
            classidx % len(bbox_colors)
        ]


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (xmin, ymin),
            (xmax, ymax),
            color,
            3
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{classname}: "
            f"{conf * 100:.1f}%"
        )


        label_size, base_line = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )
        )


        label_ymin = max(
            ymin,
            label_size[1] + 10
        )


        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (
                xmin,
                label_ymin
                - label_size[1]
                - 10
            ),
            (
                xmin
                + label_size[0]
                + 8,
                label_ymin
                + base_line
                - 8
            ),
            color,
            cv2.FILLED
        )


        # ----------------------------------------------------
        # Label text
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (
                xmin + 4,
                label_ymin - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # COUNT TYPES
    # ========================================================

    defect_counts = Counter(
        detected_types
    )


    # ========================================================
    # FPS
    # ========================================================

    t_stop = time.perf_counter()

    elapsed = (
        t_stop - t_start
    )

    if elapsed > 0:

        current_fps = (
            1.0 / elapsed
        )

    else:

        current_fps = 0


    if len(frame_rate_buffer) >= fps_avg_len:

        frame_rate_buffer.pop(0)

    frame_rate_buffer.append(
        current_fps
    )

    avg_frame_rate = np.mean(
        frame_rate_buffer
    )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    panel_x = 15
    panel_y = 20

    line_height = 30


    # --------------------------------------------------------
    # Total detections
    # --------------------------------------------------------

    total_text = (
        f"Total Detections: "
        f"{object_count}"
    )

    cv2.putText(
        frame,
        total_text,
        (
            panel_x,
            panel_y + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # DEFECT TYPES
    # ========================================================

    cv2.putText(
        frame,
        "Defect Types:",
        (
            panel_x,
            panel_y + 65
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Display each defect type
    # --------------------------------------------------------

    current_y = (
        panel_y + 95
    )


    if object_count == 0:

        cv2.putText(
            frame,
            "No defects detected",
            (
                panel_x,
                current_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        for defect_type, count in (
            defect_counts.items()
        ):

            type_text = (
                f"{defect_type}: "
                f"{count}"
            )

            cv2.putText(
                frame,
                type_text,
                (
                    panel_x,
                    current_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            current_y += line_height


    # ========================================================
    # FPS
    # ========================================================

    if source_type in [
        "video",
        "usb"
    ]:

        fps_text = (
            f"FPS: "
            f"{avg_frame_rate:.2f}"
        )

        cv2.putText(
            frame,
            fps_text,
            (
                panel_x,
                current_y + 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # AUTOMATIC DISPLAY RESIZING
    # ========================================================

    if user_res_enabled:

        display_frame = cv2.resize(
            frame,
            (
                resW,
                resH
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        display_frame = fit_to_screen(
            frame,
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # RECORD
    # ========================================================

    if record:

        recorded_frame = cv2.resize(
            frame,
            (
                resW,
                resH
            ),
            interpolation=cv2.INTER_AREA
        )

        recorder.write(
            recorded_frame
        )


    # ========================================================
    # KEYBOARD
    # ========================================================

    if source_type in [
        "image",
        "folder"
    ]:

        key = cv2.waitKey(0)

    else:

        key = cv2.waitKey(5)


    # Q = Quit
    if key in [
        ord("q"),
        ord("Q")
    ]:

        break


    # S = Pause
    elif key in [
        ord("s"),
        ord("S")
    ]:

        print(
            "Inference paused. "
            "Press any key to continue."
        )

        cv2.waitKey(0)


    # P = Save
    elif key in [
        ord("p"),
        ord("P")
    ]:

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"detection_{timestamp}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Detection saved as: "
            f"{filename}"
        )


# ============================================================
# CLEANUP
# ============================================================

print(
    f"\nAverage pipeline FPS: "
    f"{avg_frame_rate:.2f}"
)


if source_type in [
    "video",
    "usb"
]:

    cap.release()


if record:

    recorder.release()

    print(
        f"Recorded video saved as: "
        f"{record_name}"
    )


cv2.destroyAllWindows()

print(
    "\nProgram closed successfully."
)