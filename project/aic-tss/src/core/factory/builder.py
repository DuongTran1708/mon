#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory classes.
"""

from __future__ import annotations

from .factory import Factory

__all__ = [
	"CAMERAS",
	"DETECTORS",
	"IDENTIFICATIONS",

	"FILE_HANDLERS"
]

# MARK: - Modules

CAMERAS         = Factory(name="cameras")
DETECTORS       = Factory(name="object_detectors")
IDENTIFICATIONS = Factory(name="identifications")

# MARK: - File

FILE_HANDLERS   = Factory(name="file_handler")
