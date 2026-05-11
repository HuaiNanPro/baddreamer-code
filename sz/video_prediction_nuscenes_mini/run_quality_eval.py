#!/usr/bin/env python
"""
Wrapper script to run quality_evaluation.py with correct PYTHONPATH
"""
import os
import sys

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create a virtual module name for backwards compatibility
# This allows 'from video_prediction_nuscenes_mini.vam.datalib import ...' to work
sys.modules['video_prediction_nuscenes_mini'] = sys.modules[__name__]

# Now import and run the main script
from scripts import quality_evaluation