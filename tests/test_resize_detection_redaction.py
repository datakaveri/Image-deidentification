import numpy as np

from app.deidentification.resize_stage import (
    map_human_mask_to_original,
    map_plate_detections_to_original,
    map_watermark_regions_to_original,
    prepare_detection_image,
)
from app.deidentification.contracts import PlateDetection


def test_prepare_detection_image_scales_down_large_images():
    image = np.zeros((4000, 3000, 3), dtype=np.uint8)
    resized, scale_x, scale_y = prepare_detection_image(image, max_side=1600)

    assert resized.shape[0] <= 1600
    assert resized.shape[1] <= 1600
    assert 0 < scale_x < 1
    assert 0 < scale_y < 1


def test_detection_mapping_uses_original_coordinates():
    original = np.zeros((300, 400, 3), dtype=np.uint8)
    resized, scale_x, scale_y = prepare_detection_image(original, max_side=150)

    plate = PlateDetection((10, 20, 30, 40), "ABC123", 0.9, "plate")
    mapped = map_plate_detections_to_original([plate], scale_x, scale_y)
    assert mapped[0].box == (int(10 / scale_x), int(20 / scale_y), int(30 / scale_x), int(40 / scale_y))

    mask = np.zeros((resized.shape[0], resized.shape[1]), dtype=np.uint8)
    mask[5:15, 10:20] = 255
    mapped_mask = map_human_mask_to_original(mask, original.shape[:2], scale_x, scale_y)
    assert mapped_mask.shape == original.shape[:2]
    assert mapped_mask.max() == 255

    regions = [(10, 20, 30, 40)]
    mapped_regions = map_watermark_regions_to_original(regions, scale_x, scale_y)
    assert mapped_regions[0][0] > 10
    assert mapped_regions[0][2] > mapped_regions[0][0]
