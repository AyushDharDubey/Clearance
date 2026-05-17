# ======================================
# Intel Labs MiDaS
# Courtesy : Ultralytics
# ======================================

import cv2
import torch
import numpy as np


class MiDaS:
    """Performs monocular depth estimation using Intel Labs MiDaS models.

    This class provides utilities to load a pre-trained MiDaS model,
    apply image transforms, generate depth maps, and normalize the results
    for visualization. It also supports real-time depth inference from video streams.

    Attributes:
        midas (torch.nn.Module): The MiDaS model instance.
        transform (callable): The preprocessing transform for input images.
        model_type (str): The type of MiDaS model to load.
        device (torch.device): The computation device (CPU or CUDA).
    """

    def __init__(self, model_type: str):
        """Initializes the MiDaS depth estimation class.

        Args:
            model_type (str): The model variant to use.
                Supported values:
                - "DPT_Large": Highest accuracy, slowest speed.
                - "DPT_Hybrid": Balanced accuracy and speed.
                - "MiDaS_small": Fastest, lowest accuracy.
        """
        self.midas = None
        self.transform = None
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self) -> None:
        """Loads the MiDaS model from the PyTorch Hub."""
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device).eval()

    def transforms(self):
        """Retrieves the appropriate image preprocessing transform."""
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        return self.transform

    def depth_map(self, batch: torch.Tensor, img: np.ndarray) -> np.ndarray:
        """Generates a depth map for a given input image batch."""
        with torch.no_grad():
            prediction = self.midas(batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    @staticmethod
    def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
        """Normalizes and colorizes a depth map for visualization."""
        depth_min, depth_max = depth_map.min(), depth_map.max()
        normalized = (depth_map - depth_min) / (depth_max - depth_min)
        normalized = (normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

    @staticmethod
    def stack_frames(frame1: np.ndarray, frame2: np.ndarray, width: int = 1280, height: int = 640) -> np.ndarray:
        """Safely stacks two frames horizontally with resizing to a fixed display resolution."""
        h, w = frame1.shape[:2]
        frame2 = cv2.resize(frame2, (w, h))
        combined = np.hstack((frame1, frame2))
        combined = cv2.resize(combined, (width, height))
        return combined

    def infer_video(self, source: str = 0, output_path: str = None, display: bool = True) -> None:
        """Performs real-time depth estimation on a video stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        width = 1280
        height = 640
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        transform = self.transforms()

        print("🚀 Starting video depth inference... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(rgb).to(self.device)

            depth = self.depth_map(input_batch, rgb)
            colored_depth = self.normalize_depth(depth)
            combined = self.stack_frames(frame, colored_depth, width, height)

            if display:
                cv2.imshow("MiDaS Depth Estimation (Press 'q' to exit)", combined)

            if writer:
                writer.write(combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Inference completed and resources released.")


if __name__ == "__main__":
    """Example usage for both image and video depth estimation."""
    # midas = MiDaS(model_type="MiDaS_small")
    midas = MiDaS(model_type="DPT_Hybrid")
    # midas = MiDaS(model_type="DPT_Large")
    

    # ---------- Image inference ----------
    filename = "videos/d14_f002220.jpg"
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    transform = midas.transforms()
    input_batch = transform(img).to(midas.device)
    
    d_map = midas.depth_map(batch=input_batch, img=img)
    normalize_d_map = midas.normalize_depth(d_map)
    cv2.imwrite("depth_colored.png", normalize_d_map)

    # ---------- Video inference ----------
    # For webcam: source=0
    # For file:   source="cars.mp4"
    # midas.infer_video(source="videos/cars.mp4", output_path="depth_video.mp4", display=True)
