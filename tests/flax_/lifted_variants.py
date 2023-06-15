import enum
import inspect
from typing import Type
from types import MethodType

import flax.linen as nn
import toolz
from chex._src.variants import (  # noqa
    Variant,
    check_variant_arguments,
    VariantsTestCaseGenerator,
    _variant_decorators,  # noqa
    _valid_kwargs_keys,  # noqa
)


class LiftedVariantType(enum.Enum):
    WITH_LIFTED_JIT = enum.auto()
    WITHOUT_LIFTED_JIT = enum.auto()

    def __str__(self) -> str:
        return "_" + self.name.lower()


@toolz.curry
def _variants_fn(
    test_object: MethodType, **which_variants
) -> VariantsTestCaseGenerator:
    """Implements `variants` and `all_variants`."""

    # Convert keys to enum entries.
    which_variants = {
        LiftedVariantType[name.upper()]: var for name, var in which_variants.items()
    }
    if isinstance(test_object, VariantsTestCaseGenerator):
        # Merge variants for nested wrappers.
        test_object.add_variants(which_variants)
    else:
        test_object = VariantsTestCaseGenerator(test_object, which_variants)

    return test_object


@toolz.curry
def variants(
    test_method: MethodType,
    with_lifted_jit: bool = False,  # noqa
    without_lifted_jit: bool = False,  # noqa
) -> VariantsTestCaseGenerator:
    return _variants_fn(
        test_method,
        with_lifted_jit=with_lifted_jit,
        without_lifted_jit=without_lifted_jit,
    )


@toolz.curry
@check_variant_arguments
def _with_lifted_jit(target: Type[nn.Module], *args, **kwargs) -> Type[nn.Module]:
    return nn.jit(target, *args, **kwargs)


@toolz.curry
@check_variant_arguments
def _without_lifted_jit(target: Type[nn.Module]) -> Type[nn.Module]:
    return target


_variant_decorators.update(
    dict(
        {
            LiftedVariantType.WITH_LIFTED_JIT: _with_lifted_jit,
            LiftedVariantType.WITHOUT_LIFTED_JIT: _without_lifted_jit,
        }
    )
)

# Expose variant objects.
without_lifted_jit = Variant("without_lifted_jit", _without_lifted_jit)
with_lifted_jit = Variant("with_lifted_jit", _with_lifted_jit)
ALL_VARIANTS = (without_lifted_jit, with_lifted_jit)
# Collect valid argument names from all variant decorators.
for fn_ in _variant_decorators.values():
    original_fn = fn_.func.__wrapped__
    _valid_kwargs_keys.update(inspect.getfullargspec(original_fn).args)
