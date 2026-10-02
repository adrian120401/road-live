"""Check automatic CPU fallback without requiring an NVIDIA GPU."""

import unittest
from unittest.mock import patch

from src.tracker import select_device


class DeviceTests(unittest.TestCase):
    def test_explicit_cpu_does_not_probe_cuda(self):
        with patch("src.tracker.torch.cuda.is_available") as available:
            self.assertEqual(select_device("cpu"), "cpu")
            available.assert_not_called()

    def test_auto_falls_back_without_cuda(self):
        with patch("src.tracker.torch.cuda.is_available", return_value=False):
            with self.assertLogs("src.tracker", level="WARNING"):
                self.assertEqual(select_device("auto"), "cpu")

    def test_auto_falls_back_if_cuda_kernel_is_incompatible(self):
        with patch("src.tracker.torch.cuda.is_available", return_value=True), \
             patch("src.tracker.torch.ones", side_effect=RuntimeError("no kernel image")):
            with self.assertLogs("src.tracker", level="WARNING"):
                self.assertEqual(select_device("auto"), "cpu")

    def test_explicit_cuda_reports_failure(self):
        with patch("src.tracker.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                select_device("cuda")
