
This showed up in the ComfyUI error log:

E:\AI\Windows-ComfyUI\custom_nodes\comfyui-sbs-long-video\nodes.py:299: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:209.)

This should be investigated and possibly addressed, if necessary.

It only showed up once in the logs and so, perhaps, is not a concern...Or perhaps it only showed up once because it's a warning?