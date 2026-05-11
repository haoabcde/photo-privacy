import unittest
from unittest.mock import patch
import numpy as np
import face_processor


class FaceProcessorTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((1200, 800, 3), dtype=np.uint8)

    def test_select_primary_face_prefers_large_center_face(self):
        boxes = [
            (40, 80, 80, 90),
            (260, 180, 210, 250),
            (640, 760, 70, 75),
        ]
        selected = face_processor._select_primary_face(boxes, self.image.shape)
        self.assertEqual(selected, [(260, 180, 210, 250)])

    def test_single_detect_mode_ignores_extra_candidates(self):
        yunet_boxes = [(270, 180, 220, 240)]
        haar_boxes = [(30, 40, 70, 80), (600, 900, 90, 90)]
        with patch.object(face_processor, '_detect_yunet', return_value=yunet_boxes), \
             patch.object(face_processor, '_detect_haar', return_value=haar_boxes):
            faces = face_processor.detect_faces(self.image, mode='blur', detect_mode='single')
        self.assertEqual(len(faces), 1)
        x, y, w, h = faces[0]
        self.assertTrue(x <= 270 and y <= 180)
        self.assertTrue(w >= 220 and h >= 240)

    def test_multi_detect_mode_can_keep_multiple_faces(self):
        yunet_boxes = [(100, 100, 120, 140), (420, 130, 110, 135)]
        with patch.object(face_processor, '_detect_yunet', return_value=yunet_boxes), \
             patch.object(face_processor, '_detect_haar', return_value=[]):
            faces = face_processor.detect_faces(self.image, mode='avatar', detect_mode='multi')
        self.assertEqual(len(faces), 2)

    # Test removed as _resolve_detect_mode no longer exists

    def test_strict_mode_disables_haar(self):
        yunet_boxes = [] # Empty to normally trigger Haar fallback
        haar_boxes = [(30, 40, 70, 80)]
        
        with patch.object(face_processor, '_detect_yunet', return_value=yunet_boxes) as mock_yunet, \
             patch.object(face_processor, '_detect_haar', return_value=haar_boxes) as mock_haar:
            
            # Non-strict should call Haar and find 1 face
            faces_normal = face_processor.detect_faces(self.image, mode='blur', detect_mode='multi', strict=False)
            self.assertEqual(len(faces_normal), 1)
            mock_haar.assert_called()
            
            mock_haar.reset_mock()
            
            # Strict should NOT call Haar and find 0 faces
            faces_strict = face_processor.detect_faces(self.image, mode='blur', detect_mode='multi', strict=True)
            self.assertEqual(len(faces_strict), 0)
            mock_haar.assert_not_called()

    def test_process_image_honors_single_detect_mode(self):
        with patch.object(face_processor, 'correct_image_orientation', return_value=self.image), \
             patch.object(face_processor, 'detect_faces', return_value=[]) as mock_detect:
            face_processor.process_image(b'image-bytes', mode='blur', detect_mode='single')

        self.assertEqual(mock_detect.call_args.kwargs['detect_mode'], 'single')

    def test_process_preview_honors_single_detect_mode(self):
        with patch.object(face_processor, 'correct_image_orientation', return_value=self.image), \
             patch.object(face_processor, 'detect_faces', return_value=[]) as mock_detect:
            face_processor.process_preview(b'image-bytes', mode='blur', detect_mode='single')

        self.assertEqual(mock_detect.call_args.kwargs['detect_mode'], 'single')

if __name__ == '__main__':
    unittest.main()
