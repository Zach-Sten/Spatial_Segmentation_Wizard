"""
config_loader.py — Load config, discover samples, resolve output paths.

Supports three data modes:
  1. experiment_dir → crawl all slides, find all output-* samples
  2. slide_dir      → find all output-* samples in one slide
  3. sample_dir     → single output-* folder

Each discovered sample becomes a SampleInfo with resolved input/output paths.
"""

import os
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# What can actually run
# ============================================================================
#
# Scaffolding exists for these, but they are not validated end to end. They are
# shown in the wizard as "(coming soon)" and cannot be selected or enabled: a
# half-working method that silently writes plausible output is far worse than
# one that isn't offered.
#
# To promote a method: delete it from this set, confirm it has an entry in
# METHOD_SCRIPTS (scripts/slurm/generate_slurm.py), and give it defaults in the
# wizard's METHOD_DEFAULTS.
COMING_SOON = {"bidcell", "bering", "segger"}

# Platforms the loader can actually read. Same rule as methods — the wizard
# offers the rest as "(coming soon)" rather than letting a run fail at load time.
SUPPORTED_PLATFORMS = {"xenium"}
COMING_SOON_PLATFORMS = ["cosmx", "merscope", "stereoseq"]


# ============================================================================
# SampleInfo — one discovered sample with all paths resolved
# ============================================================================

@dataclass
class SampleInfo:
    """Represents a single discovered sample ready for processing."""
    sample_id: str              # e.g. "XETG00143__0032645"
    sample_dir: Path            # full path to the output-XETG... folder (raw input)
    slide_dir: Path             # parent slide folder
    slide_name: str             # e.g. "20241114__203842__11142024_SPITZER_HN_DYSPLASIA1"
    platform: str               # e.g. "xenium"

    def output_dir(self, method: str, output_base_override: str = "") -> Path:
        """
        Where results go for this sample + method.

        Default: {slide_dir}/{method}_reseg/{sample_id}/
        Override: {output_base_override}/{slide_name}/{method}_reseg/{sample_id}/
        """
        if output_base_override:
            return Path(output_base_override) / self.slide_name / f"{method}_reseg" / self.sample_id
        return self.slide_dir / f"{method}_reseg" / self.sample_id

    def log_dir_in_pipeline(self, pipeline_root: str) -> Path:
        """Logs inside the pipeline directory (always writable)."""
        return Path(pipeline_root) / "logs" / self.slide_name


# ============================================================================
# Site config — cluster-specific settings, see config/site.yaml
# ============================================================================

# Defaults describe the cluster this pipeline was developed on. Any site.yaml
# key overrides its entry here; anything absent falls back to these values, so
# a missing or partial site.yaml still produces a working script.
SITE_DEFAULTS = {
    "container_cmd": "singularity",
    "preamble":      [],
    # Absolute, not bare `python`: the container's %environment prepends
    # /opt/miniforge3/bin (base conda) to PATH *after* activating the env, so
    # bare `python` resolves to the base interpreter with none of the
    # segmentation packages installed.
    "python_bin":    "/opt/miniforge3/envs/spatial_segmentation_env/bin/python",
    "gpu_gres":      "gpu:1g.10gb:1",
    "gpu_partition": "common",
    "gpu_account":   "common",
    "extra_binds":   [],
    "ulimit_nofile": 65536,
}

_site_cache = None
_warned_containers = set()


def load_site_config(pipeline_root: str = None) -> dict:
    """Return site settings: SITE_DEFAULTS overlaid with config/site.yaml.

    Looked up in order: $SEGWIZ_SITE_CONFIG, then {pipeline_root}/config/site.yaml.
    Result is cached — the site does not change within a run.
    """
    global _site_cache
    if _site_cache is not None:
        return _site_cache

    if pipeline_root is None:
        pipeline_root = Path(__file__).resolve().parents[2]

    site_path = os.environ.get("SEGWIZ_SITE_CONFIG") or Path(pipeline_root) / "config" / "site.yaml"
    site = dict(SITE_DEFAULTS)

    if Path(site_path).exists():
        with open(site_path) as f:
            loaded = yaml.safe_load(f) or {}
        unknown = set(loaded) - set(SITE_DEFAULTS)
        if unknown:
            # Typos here silently revert to a default and produce a script that
            # fails minutes into a job, so say so at generation time instead.
            print(f"[WARN] Ignoring unknown key(s) in {site_path}: {', '.join(sorted(unknown))}")
        site.update({k: v for k, v in loaded.items() if k in SITE_DEFAULTS})
    else:
        print(f"[INFO] No site.yaml at {site_path} — using built-in cluster defaults")

    _site_cache = site
    return site


# ============================================================================
# Config loading
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load and validate the pipeline YAML config."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Validate required sections
    if "data" not in cfg:
        raise ValueError("Config missing 'data' section")
    if "paths" not in cfg:
        raise ValueError("Config missing 'paths' section")
    if "container_sif" not in cfg["paths"]:
        raise ValueError("Config missing paths.container_sif")

    data = cfg["data"]

    platform = data.get("platform", "xenium")
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform '{platform}' is not supported yet. "
            f"Currently available: {', '.join(sorted(SUPPORTED_PLATFORMS))}."
        )

    has_experiment = bool(data.get("experiment_dir"))
    has_slide = bool(data.get("slide_dir"))
    has_sample = bool(data.get("sample_dir"))

    active_modes = sum([has_experiment, has_slide, has_sample])
    if active_modes == 0:
        raise ValueError(
            "Config data section must set one of: experiment_dir, slide_dir, or sample_dir"
        )
    if active_modes > 1:
        raise ValueError(
            "Config data section has multiple modes set. "
            "Use exactly ONE of: experiment_dir, slide_dir, or sample_dir"
        )

    return cfg


# ============================================================================
# Sample discovery
# ============================================================================

def _extract_sample_id(folder_name: str) -> str:
    """
    Extract a clean sample ID from an output-* folder name.

    'output-XETG00143__0032645__Region_1__20241114__203854'
    →  'XETG00143__0032645'

    Falls back to the full folder name if the pattern doesn't match.
    """
    # Strip the leading "output-"
    name = folder_name
    if name.startswith("output-"):
        name = name[len("output-"):]

    # Try to extract the cartridge + slide portion: XETG00143__0032645
    # Pattern: letters+digits + __ + digits (the two primary identifiers)
    match = re.match(r"^([A-Za-z0-9]+__\d+)", name)
    if match:
        return match.group(1)

    # Fallback: return everything before __Region or just the full name
    if "__Region" in name:
        return name.split("__Region")[0]

    return name


def _matches_filters(folder_name: str, include: list, exclude: list) -> bool:
    """Check if a sample folder passes include/exclude filters."""
    if exclude:
        for pattern in exclude:
            if pattern in folder_name:
                return False
    if include:
        return any(pattern in folder_name for pattern in include)
    return True  # no include filter = accept all


def discover_samples(cfg: dict) -> List[SampleInfo]:
    """
    Discover all samples based on the config's data mode.

    Returns a sorted list of SampleInfo objects.
    """
    data = cfg["data"]
    platform = data.get("platform", "xenium")
    sample_glob = data.get("sample_glob", "output-*")
    include = data.get("include", []) or []
    exclude = data.get("exclude", []) or []

    samples = []

    if data.get("experiment_dir"):
        # Mode 1: Experiment — find all slides, then all samples in each
        exp_dir = Path(data["experiment_dir"]).resolve()
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

        # Slides are direct subdirectories of the experiment dir
        slide_dirs = sorted([
            d for d in exp_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not slide_dirs:
            raise ValueError(f"No slide folders found in: {exp_dir}")

        for slide_dir in slide_dirs:
            sample_dirs = sorted(slide_dir.glob(sample_glob))
            for sd in sample_dirs:
                if sd.is_dir() and _matches_filters(sd.name, include, exclude):
                    samples.append(SampleInfo(
                        sample_id=_extract_sample_id(sd.name),
                        sample_dir=sd.resolve(),
                        slide_dir=slide_dir.resolve(),
                        slide_name=slide_dir.name,
                        platform=platform,
                    ))

    elif data.get("slide_dir"):
        # Mode 2: Slide — direct child folders that look like raw samples
        slide_dir = Path(data["slide_dir"]).resolve()
        if not slide_dir.exists():
            raise FileNotFoundError(f"Slide dir not found: {slide_dir}")

        # Directories to skip — output/cache dirs are never raw samples
        _skip_suffixes = ("_reseg",)
        _skip_names    = {"qc", "logs"}
        _skip_prefixes = ("classifier_cache", "cellspa_qc")

        for sd in sorted(slide_dir.iterdir()):
            if not sd.is_dir() or sd.name.startswith("."):
                continue
            if (sd.name.endswith(_skip_suffixes)
                    or sd.name in _skip_names
                    or any(sd.name.startswith(p) for p in _skip_prefixes)):
                continue
            # Recognize xenium samples by the presence of experiment.xenium
            # (fall back to cell_feature_matrix/ for other platforms)
            if not ((sd / "experiment.xenium").exists() or (sd / "cell_feature_matrix").exists()):
                continue
            if _matches_filters(sd.name, include, exclude):
                samples.append(SampleInfo(
                    sample_id=sd.name,
                    sample_dir=sd.resolve(),
                    slide_dir=slide_dir,
                    slide_name=slide_dir.name,
                    platform=platform,
                ))

    elif data.get("sample_dir"):
        # Single sample
        sample_dir = Path(data["sample_dir"]).resolve()
        if not sample_dir.exists():
            raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

        slide_dir = sample_dir.parent
        samples.append(SampleInfo(
            sample_id=_extract_sample_id(sample_dir.name),
            sample_dir=sample_dir,
            slide_dir=slide_dir,
            slide_name=slide_dir.name,
            platform=platform,
        ))

    if not samples:
        raise ValueError(
            "No samples discovered. Check your data paths and include/exclude filters."
        )

    return samples


# ============================================================================
# Config accessors (method config, container path, etc.)
# ============================================================================

def get_method_config(cfg: dict, method: str) -> dict:
    """Get merged SLURM + params config for a specific method."""
    methods = cfg.get("methods", {})
    if method not in methods:
        raise ValueError(f"Method '{method}' not found in config. Available: {list(methods.keys())}")

    method_cfg = methods[method]

    # Merge SLURM: method overrides → defaults
    slurm_defaults = cfg.get("slurm", {}).get("default", {})
    slurm_method = method_cfg.get("slurm", {})
    merged_slurm = {**slurm_defaults, **slurm_method}

    # Add global SLURM settings
    for key in ["partition", "account", "email", "mail_type"]:
        if key in cfg.get("slurm", {}):
            merged_slurm[key] = cfg["slurm"][key]

    return {
        "enabled": method_cfg.get("enabled", True),
        "slurm": merged_slurm,
        "params": method_cfg.get("params", {}),
    }


def get_output_base_override(cfg: dict) -> str:
    """Get the output base override path (empty string = use default layout)."""
    return cfg.get("paths", {}).get("output_base_override", "") or ""


def get_container_path(cfg: dict) -> str:
    """Return the .sif container path, resolved to absolute.

    Only warns if missing — generating scripts on a machine without the
    container is a supported dry-run workflow. Submitting with a bad path is
    not; check_container_exists() enforces that at submit time.
    """
    container = Path(cfg["paths"]["container_sif"]).resolve()
    # Called once per generated script; warn on the first miss only.
    if not container.exists() and container not in _warned_containers:
        _warned_containers.add(container)
        print(f"[WARN] Container not found (fine for a dry run): {container}")
    return str(container)


def check_container_exists(cfg: dict):
    """Hard-fail before submitting. Without this, a stale path in the config
    submits every job successfully and each one dies seconds later."""
    container = Path(cfg["paths"]["container_sif"]).resolve()
    if not container.exists():
        raise FileNotFoundError(
            f"Container not found: {container}\n"
            f"Fix paths.container_sif in your config before submitting."
        )


def list_enabled_methods(cfg: dict) -> list:
    """Return enabled methods, minus any that aren't runnable yet.

    Filtering here rather than in the wizard means a hand-edited config cannot
    slip a coming-soon method past the interactive menu.
    """
    enabled = [
        name for name, mcfg in cfg.get("methods", {}).items()
        if mcfg.get("enabled", True)
    ]
    blocked = [m for m in enabled if m in COMING_SOON]
    if blocked:
        print(f"[WARN] Not yet available, skipping: {', '.join(sorted(blocked))}")
    return [m for m in enabled if m not in COMING_SOON]


def is_slide_mode(cfg: dict) -> bool:
    """True when config uses slide_dir mode (multiple samples in one folder)."""
    return bool(cfg.get("data", {}).get("slide_dir"))
