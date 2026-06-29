from src.features.manual_tabular.preprocessors.alphamissense import AlphaMissensePreprocessor
from src.features.manual_tabular.preprocessors.charge import ChargeFeaturePreprocessor
from src.features.manual_tabular.preprocessors.domain import DomainPreprocessor
from src.features.manual_tabular.preprocessors.idr import IDRRegionPreprocessor
from src.features.manual_tabular.preprocessors.kinase import KinasePriorPreprocessor
from src.features.manual_tabular.preprocessors.motif1433 import Motif1433Preprocessor
from src.features.manual_tabular.preprocessors.nes import NESFeaturePreprocessor
from src.features.manual_tabular.preprocessors.nls import NLSFeaturePreprocessor
from src.features.manual_tabular.preprocessors.sequence import SequencePhysicochemPreprocessor
from src.features.manual_tabular.preprocessors.hotspot import HotspotPreprocessor
from src.features.manual_tabular.preprocessors.zscale import ZScalePreprocessor
from src.features.manual_tabular.preprocessors.evolution import EvolutionPreprocessor

FAMILY_REGISTRY = {
    "charge": {
        "selector": lambda df: [c for c in df.columns if c.startswith("SEQ_Charge_")],
        "preprocessor_cls": ChargeFeaturePreprocessor,
        "kwargs_key": "charge_kwargs",
    },
    "sequence": {
        "selector": lambda df: [
            c for c in df.columns
            if c.startswith("SEQ_")
            and not c.startswith("SEQ_Charge_")
            and not c.startswith("SEQ_Hotspot_")
            and not c.startswith("SEQ_ZScale_")
        ],
        "preprocessor_cls": SequencePhysicochemPreprocessor,
        "kwargs_key": "sequence_kwargs",
    },
    "zscale": {
        "selector": lambda df: [c for c in df.columns if c.startswith("SEQ_ZScale_")],
        "preprocessor_cls": ZScalePreprocessor,
        "kwargs_key": "zscale_kwargs",
    },
    "hotspot": {
        "selector": lambda df: [c for c in df.columns if c.startswith("SEQ_Hotspot_")],
        "preprocessor_cls": HotspotPreprocessor,
        "kwargs_key": "hotspot_kwargs",
    },
    "kinase": {
        "selector": lambda df: [c for c in df.columns if c.startswith("FUNC_Kinase_")],
        "preprocessor_cls": KinasePriorPreprocessor,
        "kwargs_key": "kinase_kwargs",
    },
    "alphamissense": {
        "selector": lambda df: [c for c in df.columns if c.startswith("FUNC_AlphaMissense_")],
        "preprocessor_cls": AlphaMissensePreprocessor,
        "kwargs_key": "alphamissense_kwargs",
    },
    "nls": {
        "selector": lambda df: [c for c in df.columns if c.startswith("MOTIF_NLS_")],
        "preprocessor_cls": NLSFeaturePreprocessor,
        "kwargs_key": "nls_kwargs",
    },
    "nes": {
        "selector": lambda df: [c for c in df.columns if c.startswith("MOTIF_NES_")],
        "preprocessor_cls": NESFeaturePreprocessor,
        "kwargs_key": "nes_kwargs",
    },
    "motif1433": {
        "selector": lambda df: [c for c in df.columns if c.startswith("MOTIF_1433_")],
        "preprocessor_cls": Motif1433Preprocessor,
        "kwargs_key": "motif1433_kwargs",
    },
    "idr": {
        "selector": lambda df: [c for c in df.columns if c.startswith("MOTIF_IDR_")],
        "preprocessor_cls": IDRRegionPreprocessor,
        "kwargs_key": "idr_kwargs",
    },
    "domain": {
        "selector": lambda df: [c for c in df.columns if c.startswith("MOTIF_Domain_")],
        "preprocessor_cls": DomainPreprocessor,
        "kwargs_key": "domain_kwargs",
    },
    "evolution": {
        "selector": lambda df: [c for c in df.columns if c.startswith("FUNC_Evolution_")],
        "preprocessor_cls": EvolutionPreprocessor,
        "kwargs_key": "evolution_kwargs",
    },
}