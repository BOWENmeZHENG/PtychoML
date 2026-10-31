"""Subclasses of cdtools models with modified behavior."""

from cdtools.models import FancyPtycho


class MLAugmentedPtycho(FancyPtycho):
    """FancyPtycho with overrides.

    Methods defined on the class (``interaction``, ``measurement``,
    ``forward_propagator``, ...) can be overridden normally.
    """
