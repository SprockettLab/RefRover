"""Binner formatter registry."""

from .generic import write_generic
from .metabat2 import write_metabat2
from .semibin2 import write_semibin2
from .maxbin2 import write_maxbin2
from .concoct import write_concoct

BINNER_REGISTRY = {
    "generic": write_generic,
    "metabat2": write_metabat2,
    "semibin2": write_semibin2,
    "maxbin2": write_maxbin2,
    "concoct": write_concoct,
}
