"""Inference-only stand-ins for optional packages that xLLM imports at module load (kernels, training and
evaluation tooling). Any attribute of a stubbed module is a placeholder that raises when called, so a code
path that really needs one of these packages fails loudly instead of running something different."""
import importlib.abc
import importlib.machinery
import sys
import types

STUBBED = {"fla", "flash_attn", "flash_attn_3", "xattn", "evalplus", "interfering", "nltk", "pynvml",
           "rouge_score", "submitit"}


class _Missing:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise NotImplementedError(f"{self._name} is stubbed for inference")

    def __getattr__(self, attr):
        return _Missing(f"{self._name}.{attr}")

    def __mro_entries__(self, bases):
        return (object,)


class _StubModule(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self.__name__}.{attr}")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


sys.meta_path.append(_Finder())
