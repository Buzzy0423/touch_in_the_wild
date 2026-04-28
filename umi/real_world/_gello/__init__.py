"""Vendored copies of GELLO camera utilities used by GelloMultiUvcCamera.

These are direct copies (with relative imports) from the gello_software
repo so touch_in_the_wild can run the threaded camera path without
requiring gello to be pip-installed in the touchwild env.

Source files:
- elgato_reset.py  <- gello/cameras/elgato_reset.py
- ring_buffer.py   <- gello/shared_memory/ring_buffer.py
- uvc_camera.py    <- gello/cameras/uvc_camera.py
"""
