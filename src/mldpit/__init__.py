"""MLdpit — deep learning temporal downscaling of precipitation.

Converts hourly precipitation fields into 10-minute fields using a U-Net with a
softmax mass-conservation constraint, so that the six predicted sub-hourly
fields sum exactly to the hourly total they were derived from.
"""

__version__ = "0.9.0"

__all__ = ["__version__"]
