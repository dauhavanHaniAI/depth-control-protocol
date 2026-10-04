"""Pure-PyTorch stand-in for xLLM's compiled extension, inference only.
Only the group RMSNorm forward is implemented (fp32 accumulation, as in group_rms_norm_kernel.cu);
every other op raises when called, so any unexpected use fails loudly."""
from xllm_autostub import _Missing

te = _Missing("xllm_extension.te")
