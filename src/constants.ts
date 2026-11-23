import type { ProjectModule, ProjectTask } from "./types";

export const APP_NAME = "Damnisium";
export const APP_VERSION = "v1-beta";

export const CLASS_PALETTE = [
  "#38bdf8",
  "#a855f7",
  "#22c55e",
  "#f59e0b",
  "#ef4444",
  "#ec4899",
  "#14b8a6",
  "#6366f1",
  "#eab308",
  "#06b6d4",
];

export const MODULE_TASKS: Record<
  ProjectModule,
  { label: string; tasks: { value: ProjectTask; label: string; hint: string }[] }
> = {
  image: {
    label: "Image ML",
    tasks: [
      { value: "detection", label: "Object Detection (Boxes)", hint: "YOLO bounding boxes" },
      { value: "segmentation", label: "Instance Segmentation (Polygons)", hint: "YOLO-seg polygons" },
      { value: "classification", label: "Image Classification", hint: "MobileNet / EfficientNet" },
    ],
  },
  audio: {
    label: "Audio ML",
    tasks: [
      { value: "classification", label: "Audio Event Tagging (YAMNet)", hint: "Slice & class tag clips" },
      { value: "transcription", label: "Speech Transcription (Whisper)", hint: "Slice & transcript pairs" },
    ],
  },
  text: {
    label: "Text / Q&A",
    tasks: [
      { value: "qa", label: "Question & Answer", hint: "Qwen2.5-0.5B LoRA" },
      { value: "completion", label: "Prompt & Completion", hint: "Qwen2.5-0.5B LoRA" },
      { value: "intent", label: "Intent Classification", hint: "DistilBERT classifier" },
    ],
  },
};

export const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
  "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];
