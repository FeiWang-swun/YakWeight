import os
from typing import Tuple
import joblib
import cv2
import numpy as np
from ultralytics import YOLO
from BinocularRanging import correction


class YakMeasurement:
    """
    A class for measuring yak dimensions and estimating weight using stereo vision and YOLO object detection.
    """

    def __init__(self):
        """Initialize the yak measurement system"""
        self.rectify = correction.Rectify()  # Stereo image rectification
        self.yolo_model = YOLO("models/yolov8n_yak.pt", task='predict')  # Load YOLO yak detection model
        self.regression_model = joblib.load("models/regressorGPR.pkl")  # Load trained weight prediction model
        self.size_scale = 150  # Normalization factor for body measurements
        self.weight_scale = 400  # Normalization factor for weight prediction
        self.focal_length = 1680  # Camera focal length in pixels

    @staticmethod
    def rgb_to_bgr(rgb_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB color to BGR format used by OpenCV

        Args:
            rgb_color: Tuple of (red, green, blue) values (0-255)

        Returns:
            Tuple of (blue, green, red) values
        """
        return rgb_color[2], rgb_color[1], rgb_color[0]

    @staticmethod
    def draw_rectangle(img: np.ndarray, point1: Tuple[int, int], point2: Tuple[int, int],
                       color: Tuple[int, int, int], thickness: int = 3) -> None:
        """Draw rectangle on image for yak detection visualization

        Args:
            img: Input image array
            point1: Top-left coordinates of rectangle
            point2: Bottom-right coordinates of rectangle
            color: Rectangle color in BGR format
            thickness: Line thickness in pixels
        """
        cv2.rectangle(img, point1, point2, color, thickness)

    @staticmethod
    def draw_point(img: np.ndarray, center: Tuple[int, int], radius: int,
                   color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        """Draw keypoints on yak body for measurement

        Args:
            img: Input image array
            center: (x,y) coordinates of point center
            radius: Radius of point circle
            color: Point color in BGR format
            thickness: Line thickness (-1 fills circle)

        Returns:
            Image with drawn points
        """
        return cv2.circle(img, center, radius, color, thickness)

    @staticmethod
    def calculate_disparity_map(left_img: np.ndarray, right_img: np.ndarray,
                                down_scale: bool = True, sigma: float = 1.3) -> Tuple[np.ndarray, np.ndarray]:
        """Compute depth disparity map from stereo yak images

        Args:
            left_img: Left stereo camera image
            right_img: Right stereo camera image
            down_scale: Whether to process downscaled images
            sigma: Sigma parameter for WLS filter

        Returns:
            Tuple of (true disparity values, filtered disparity visualization)
        """
        if left_img.ndim == 2:
            img_channels = 1  # Grayscale
        else:
            img_channels = 3  # Color

        # Stereo Semi-Global Block Matching parameters
        num_disparities = 16 * 10  # Disparity search range
        block_size = 3  # Matched block size

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=5,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * img_channels * block_size ** 2,  # Smoothness penalty
            P2=32 * img_channels * block_size ** 2,  # Stronger smoothness penalty
            disp12MaxDiff=1,  # Maximum allowed difference
            uniquenessRatio=6,  # Margin in percentage
            speckleWindowSize=100,  # Speckle filtering
            speckleRange=2,
            preFilterCap=15,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # Disparity filtering parameters
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(20000)  # Smoothness control
        wls_filter.setSigmaColor(sigma)  # Color similarity

        if down_scale:
            # Process downscaled images for efficiency
            left_down = cv2.pyrDown(left_img)
            right_down = cv2.pyrDown(right_img)
            scale_factor = left_img.shape[1] / left_down.shape[1]  # Scaling factor

            disparity_left_half = left_matcher.compute(left_down, right_down)
            disparity_right_half = right_matcher.compute(right_down, left_down)

            # Resize back to original dimensions
            size = (left_img.shape[1], left_img.shape[0])
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = scale_factor * disparity_left
            disparity_right = scale_factor * disparity_right
        else:
            disparity_left = left_matcher.compute(left_img, right_img)
            disparity_right = right_matcher.compute(right_img, left_img)

        # Convert to actual disparity values
        true_disp_left = disparity_left.astype(np.float32) / 16.0
        true_disp_right = disparity_right.astype(np.float32) / 16.0

        # Apply weighted least squares filter
        filtered_disp = wls_filter.filter(
            np.int16(disparity_left), left_img, None, np.int16(disparity_right)
        )
        filtered_disp = cv2.normalize(
            src=filtered_disp, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX
        )
        filtered_disp = np.uint8(filtered_disp)

        return true_disp_left, filtered_disp

    @staticmethod
    def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate distance between two yak body points

        Args:
            point1: First point coordinates (x,y)
            point2: Second point coordinates (x,y)

        Returns:
            Euclidean distance in pixels
        """
        return np.sqrt(np.sum(np.square(point1 - point2)))

    def process_image(self, image_path: str) -> Tuple[np.ndarray, float, float, float, float]:
        """Process stereo image to measure yak body dimensions

        Args:
            image_path: Path to side-by-side stereo image

        Returns:
            Tuple containing:
            - Annotated image
            - Distance to yak (mm)
            - Back to tail length (cm)
            - Back to foot length (cm)
            - Tail to shoulder length (cm)
        """
        # Load and split stereo image
        img = cv2.imread(image_path)
        height, width = img.shape[0:2]
        split_pos = width // 2  # Split point for left/right images

        left_img = img[:, :split_pos, :]  # Left camera view
        right_img = img[:, split_pos:, :]  # Right camera view

        # Rectify stereo images for accurate depth calculation
        left_rect, right_rect, q_matrix = self.rectify.start(left_img, right_img)

        # Convert to grayscale for disparity processing
        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # Compute depth disparity map
        disparity_map, _ = self.calculate_disparity_map(gray_left, gray_right)

        # Generate 3D point cloud from disparity
        point_cloud = cv2.reprojectImageTo3D(disparity_map, q_matrix)

        # Detect yak using YOLO model
        results = self.yolo_model(source=left_rect, save=False)

        # Process detection results
        for result in results:
            # Find largest detection (primary yak)
            max_area = 0
            best_box = None
            best_idx = 0

            for idx, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                area = (int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))
                if area > max_area:
                    max_area = area
                    best_idx = idx
                    best_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

            if best_box is None:
                continue  # No yak detected

            left, top, right, bottom = best_box  # Bounding box coordinates

            # Extract 3D points within yak bounding box
            x_coords = []
            y_coords = []
            z_coords = []

            for y in range(top, bottom):
                for x in range(left, right):
                    point_3d = point_cloud[y][x]
                    x_coords.append(point_3d[0])  # X coordinates
                    y_coords.append(point_3d[1])  # Y coordinates
                    z_coords.append(point_3d[2])  # Z coordinates (depth)

            # Calculate median depth distance
            distance = np.median(z_coords)

            # Draw bounding box
            box_color = self.rgb_to_bgr((178, 34, 34))  # Dark red color
            self.draw_rectangle(
                left_rect, (left, top), (right, bottom), box_color
            )

            # Calculate size scaling factor based on distance
            size_proportion = distance / self.focal_length

            # Process yak keypoints (body landmarks)
            keypoints = result.keypoints.xy.cpu().numpy()[best_idx]
            points = [
                np.array([int(keypoints[0][0]), int(keypoints[0][1])]),  # Back point
                np.array([int(keypoints[1][0]), int(keypoints[1][1])]),  # Tail point
                np.array([int(keypoints[2][0]), int(keypoints[2][1])]),  # Shoulder point
                np.array([int(keypoints[3][0]), int(keypoints[3][1])])  # Foot point
            ]

            # Draw keypoints on yak body
            for point in points:
                self.draw_point(
                    left_rect, tuple(point), 4, box_color, 7
                )

            # Calculate physical measurements (converting pixels to cm)
            back_tail = round(
                (self.calculate_euclidean_distance(points[0], points[1]) * size_proportion) / 10, 2
            )
            back_foot = round(
                (self.calculate_euclidean_distance(points[0], points[3]) * size_proportion) / 10, 2
            )
            tail_shoulder = round(
                (self.calculate_euclidean_distance(points[1], points[2]) * size_proportion) / 10, 2
            )
            print(f"Back to tail: {back_tail}cm")
            print(f"Tail to shoulder: {tail_shoulder}cm")
            print(f"Back to foot: {back_foot}cm")

            return left_rect, distance, back_tail, back_foot, tail_shoulder

        return left_rect, 0, 0, 0, 0  # Return zeros if no yak detected

    def estimate_weight(self, back_tail: float, tail_shoulder: float, back_foot: float) -> float:
        """Predict yak weight using body measurements

        Args:
            back_tail: Back-to-tail length (cm)
            tail_shoulder: Tail-to-shoulder length (cm)
            back_foot: Back-to-foot length (cm)

        Returns:
            Estimated weight in kilograms
        """
        # Normalize measurements for model input
        features = np.array([
            back_tail / self.size_scale,
            tail_shoulder / self.size_scale,
            back_foot / self.size_scale
        ]).reshape(1, -1)

        # Predict and scale weight
        return self.regression_model.predict(features)[0] * self.weight_scale


def main():
    """Main processing function for yak weight estimation"""
    processor = YakMeasurement()

    # Process sample yak image
    image_dir = "BioImgs"
    image_path = os.path.join(image_dir, "10.png")
    processed_img, distance, back_tail, back_foot, tail_shoulder = processor.process_image(image_path)

    # Estimate and print weight
    weight = processor.estimate_weight(back_tail, tail_shoulder, back_foot)
    print(f"Estimated weight: {weight}kg")


if __name__ == '__main__':
    main()