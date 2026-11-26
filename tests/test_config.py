from pathlib import Path

import pytest

from evenet_align.core.types import RunConfig
from evenet_align.experiments.config import load_config, stage_hash

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["smoke.yaml", "ttbar.yaml", "ztautau.yaml"])
def test_configs_parse(name):
    assert isinstance(load_config(CONFIGS / name), RunConfig)


def test_smoke_and_full_list_the_same_keys():
    import yaml

    keys = lambda d, p="": {p + k for k, v in d.items() for k in ([k] if not isinstance(v, dict) else [k] + [k + "." + s for s in keys(v)])}
    smoke = yaml.safe_load((CONFIGS / "smoke.yaml").read_text())
    full = yaml.safe_load((CONFIGS / "ztautau.yaml").read_text())
    assert keys(smoke) == keys(full)


def test_unknown_key_and_bad_type_rejected():
    with pytest.raises(KeyError):
        load_config(CONFIGS / "smoke.yaml", ["dgpo.candidate=4"])
    with pytest.raises(TypeError):
        load_config(CONFIGS / "smoke.yaml", ["dgpo.candidates=four"])
    with pytest.raises(ValueError):
        load_config(CONFIGS / "smoke.yaml", ["dgpo.oracle_fraction=1.0"])


def test_override_changes_hash_but_seed_does_not():
    base = load_config(CONFIGS / "smoke.yaml")
    assert stage_hash(base, "dgpo") != stage_hash(load_config(CONFIGS / "smoke.yaml", ["dgpo.beta=2.0"]), "dgpo")
    assert stage_hash(base, "dgpo") == stage_hash(load_config(CONFIGS / "smoke.yaml", ["seeds=[5]"]), "dgpo")
    assert stage_hash(base, "supervised") == stage_hash(load_config(CONFIGS / "smoke.yaml", ["dgpo.beta=2.0"]), "supervised")


def test_evenet_policy_needs_its_supervised_files():
    with pytest.raises(ValueError, match="evenet-train"):
        load_config(CONFIGS / "smoke.yaml", ["policy.name=evenet"])


def test_oracle_must_name_a_function():
    with pytest.raises(ValueError):
        load_config(CONFIGS / "smoke.yaml", ["oracle.function=not_a_path"])
