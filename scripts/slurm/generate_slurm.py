#!/usr/bin/env python3
"""
generate_slurm.py — Build SLURM script content for each sample × method combination.

One SLURM .sh file per sample per method. Each script passes the resolved
sample-dir, output-dir, and sample-id to the Python runner.

Called by segmentation_wizard.py — not a standalone entry point.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import (
    get_method_config, get_output_base_override, get_container_path,
    load_site_config, SampleInfo,
)

# Method → Python script path (relative to project root)
METHOD_SCRIPTS = {
    "proseg":        "scripts/python/run_proseg.py",
    "baysor":        "scripts/python/run_baysor.py",
    "cellpose":      "scripts/python/run_cellpose.py",
    "bidcell":       "scripts/python/run_bidcell.py",
    "fastreseg":     "scripts/python/run_fastreseg.py",
    "stardist":      "scripts/python/run_stardist.py",
    "comseg":        "scripts/python/run_comseg.py",
    "xenium_export": "scripts/python/run_xenium_export.py",
    "cellspa_qc":    "scripts/python/run_qc.py",
    "classifier":    "scripts/python/run_rough_annotation_classifer.py",
}


def _sbatch_directives(job_name, log_dir, slurm, site, defaults, include_email=True):
    """Build the #SBATCH block. Cluster-specific values come from site config."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        f"#SBATCH --time={slurm.get('time', defaults['time'])}",
        f"#SBATCH --nodes={slurm.get('nodes', 1)}",
        f"#SBATCH --ntasks={slurm.get('ntasks', 1)}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', defaults['cpus_per_task'])}",
        f"#SBATCH --mem={slurm.get('mem', defaults['mem'])}",
    ]

    gpu = slurm.get("gpu", False)
    if gpu and site["gpu_gres"]:
        lines.append(f"#SBATCH --gres={site['gpu_gres']}")

    # GPU nodes usually live in a dedicated partition, and some schedulers
    # require the account to match it. Analysis config wins over the site default.
    partition = slurm.get("partition") or (site["gpu_partition"] if gpu else None)
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    account = slurm.get("account") or (site["gpu_account"] if gpu else None)
    if account:
        lines.append(f"#SBATCH --account={account}")

    if include_email and slurm.get("email"):
        lines.append(f"#SBATCH --mail-user={slurm['email']}")
        lines.append(f"#SBATCH --mail-type={slurm.get('mail_type', 'END,FAIL')}")

    return lines


def _container_block(site, container, py_args, gpu=False, extra_binds=()):
    """Build the ulimit + preamble + container-exec block."""
    lines = []
    if site["ulimit_nofile"]:
        lines += [f"ulimit -n {site['ulimit_nofile']}", ""]
    if site["preamble"]:
        lines += list(site["preamble"]) + [""]

    binds = sorted(set(extra_binds) | set(site["extra_binds"]))
    bind_flag = " ".join(f"--bind {p}" for p in binds)
    nv_flag = "--nv " if gpu else ""

    lines += [
        f"{site['container_cmd']} exec {nv_flag}{bind_flag} \\",
        f"    {container} \\",
        py_args,
        "",
        "EXIT_CODE=$?",
        "",
        "echo \"Finished: $(date)  Exit code: $EXIT_CODE\"",
        "exit $EXIT_CODE",
    ]
    return lines


def generate_slurm_script(
    cfg: dict,
    method: str,
    sample: SampleInfo,
    config_path: str,
) -> str:
    """Generate SLURM script content for one sample × one method."""
    method_cfg = get_method_config(cfg, method)
    slurm = method_cfg["slurm"]
    container = get_container_path(cfg)
    output_base = get_output_base_override(cfg)

    output_dir = sample.output_dir(method, output_base)

    # Pipeline root — needed for cd, bind paths, and logs
    pipeline_root = str(Path(config_path).resolve().parent.parent)
    site = load_site_config(pipeline_root)

    # Logs go inside the pipeline directory (always writable)
    log_dir = sample.log_dir_in_pipeline(pipeline_root)

    job_name = f"seg_{method}_{sample.sample_id}"
    python_script = METHOD_SCRIPTS.get(method)
    if not python_script:
        raise ValueError(f"No script registered for: {method}")

    lines = _sbatch_directives(
        job_name, log_dir, slurm, site,
        defaults={"time": "7-00:00:00", "cpus_per_task": 8, "mem": "400G"},
    )

    lines += [
        "",
        f"# {method.upper()} — {sample.sample_id}",
        f"# Slide: {sample.slide_name}",
        f"# Input: {sample.sample_dir}",
        f"# Output: {output_dir}",
        "",
        f"mkdir -p {log_dir}",
        f"mkdir -p {output_dir}",
        "",
        f"# cd to pipeline root so Python imports resolve",
        f"cd {pipeline_root}",
        "",
        "echo '============================================'",
        f"echo '  {method.upper()} — {sample.sample_id}'",
        f"echo '  Slide: {sample.slide_name}'",
        "echo \"  Job: $SLURM_JOB_ID  Node: $(hostname)\"",
        "echo \"  Start: $(date)\"",
        "echo '============================================'",
        "",
    ]

    # Build bind paths — collect all unique parent directories that need to be visible
    bind_paths = set()
    bind_paths.add(str(sample.slide_dir))             # slide root (covers qc/, outputs, raw data, etc.)
    bind_paths.add(str(sample.sample_dir))            # raw data
    bind_paths.add(str(output_dir))                   # output
    bind_paths.add(str(log_dir))                      # logs
    bind_paths.add(str(Path(config_path).resolve().parent))  # config dir
    bind_paths.add(pipeline_root)                             # pipeline scripts

    if method in ("bidcell", "fastreseg"):
        ref_path_data = cfg.get("data", {}).get("reference_path", "")
        if ref_path_data:
            bind_paths.add(str(Path(ref_path_data).parent))

    if method == "stardist":
        seg_models_path = cfg.get("data", {}).get("seg_models_path", "")
        if seg_models_path:
            bind_paths.add(str(Path(seg_models_path).resolve()))

    # Build the Python command with per-sample args
    python_bin = f"{site['python_bin']} -u"

    if method == "cellspa_qc":
        ref_path_qc = cfg.get("data", {}).get("reference_path", "")
        ref_arg_qc  = f" --reference-path {ref_path_qc}" if ref_path_qc else ""
        py_args = (
            f"    {python_bin} {python_script} "
            f"--config {config_path} "
            f"--sample-id {sample.sample_id} "
            f"--slide-dir {sample.slide_dir} "
            f"--sample-dir {sample.sample_dir}"
            f"{ref_arg_qc}"
        )
    elif method == "fastreseg":
        source_method = method_cfg["params"].get("source_method", "xenium")
        if source_method == "xenium":
            source_dir = sample.output_dir("xenium_export", output_base)
        else:
            source_dir = sample.output_dir(source_method, output_base)
        ref_path_data = cfg.get("data", {}).get("reference_path", "")
        ref_col_data  = cfg.get("data", {}).get("reference_celltype_col", "cell_type")
        ref_args = (f" --reference-path {ref_path_data} --reference-celltype-col {ref_col_data}"
                    if ref_path_data else "")
        py_args = (
            f"    {python_bin} {python_script} "
            f"--config {config_path} "
            f"--sample-dir {sample.sample_dir} "
            f"--output-dir {output_dir} "
            f"--sample-id {sample.sample_id} "
            f"--source-dir {source_dir}"
            f"{ref_args}"
        )
    elif method == "bidcell":
        ref_path_data = cfg.get("data", {}).get("reference_path", "")
        ref_col_data  = cfg.get("data", {}).get("reference_celltype_col", "cell_type")
        ref_args = (f" --reference-path {ref_path_data} --reference-celltype-col {ref_col_data}"
                    if ref_path_data else "")
        py_args = (
            f"    {python_bin} {python_script} "
            f"--config {config_path} "
            f"--sample-dir {sample.sample_dir} "
            f"--output-dir {output_dir} "
            f"--sample-id {sample.sample_id}"
            f"{ref_args}"
        )
    else:
        py_args = (
            f"    {python_bin} {python_script} "
            f"--config {config_path} "
            f"--sample-dir {sample.sample_dir} "
            f"--output-dir {output_dir} "
            f"--sample-id {sample.sample_id}"
        )

    lines += _container_block(site, container, py_args,
                              gpu=slurm.get("gpu", False), extra_binds=bind_paths)

    return "\n".join(lines)


def generate_classifier_script(
    cfg: dict,
    config_path: str,
    sample_ids: list = None,
) -> str:
    """Generate ONE SLURM script that trains the classifier once on the reference
    and predicts on every *_reseg/**/*.h5ad found under data_dir — covering all
    samples and all slides in a single job.
    """
    method_cfg = get_method_config(cfg, "classifier")
    slurm = method_cfg["slurm"]
    container = get_container_path(cfg)
    output_base = get_output_base_override(cfg)

    # data_dir: root under which all *_reseg results live
    data = cfg.get("data", {})
    if output_base:
        data_dir = output_base
    elif data.get("experiment_dir"):
        data_dir = data["experiment_dir"]
    elif data.get("slide_dir"):
        data_dir = data["slide_dir"]
    else:
        # single sample mode: parent slide dir holds the *_reseg dirs
        data_dir = str(Path(data["sample_dir"]).parent)

    pipeline_root = str(Path(config_path).resolve().parent.parent)
    site = load_site_config(pipeline_root)
    log_dir = Path(pipeline_root) / "logs" / "classifier"
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = "seg_classify_all"
    python_script = METHOD_SCRIPTS["classifier"]

    lines = _sbatch_directives(
        job_name, log_dir, slurm, site,
        defaults={"time": "0-06:00:00", "cpus_per_task": 8, "mem": "100G"},
        include_email=False,
    )

    ref_path = cfg.get("data", {}).get("reference_path", "") or method_cfg["params"].get("reference_path", "")
    bind_paths = set()
    bind_paths.add(str(data_dir))
    bind_paths.add(str(log_dir))
    bind_paths.add(str(Path(config_path).resolve().parent))
    bind_paths.add(pipeline_root)
    if ref_path:
        bind_paths.add(str(Path(ref_path).parent))

    python_bin = f"{site['python_bin']} -u"
    celltype_col = cfg.get("data", {}).get("reference_celltype_col", "") or method_cfg["params"].get("reference_celltype_col", "cell_type")
    gpu_flag     = " --gpu"      if slurm.get("gpu", False)                       else ""
    retrain_flag = " --retrain"  if method_cfg["params"].get("retrain", False)     else ""
    no_rank_flag    = " --no-rank"  if not method_cfg["params"].get("use_rank", True) else ""
    sample_ids_flag = (" --sample-ids " + " ".join(sample_ids)) if sample_ids else ""

    py_args = (
        f"    {python_bin} {python_script} "
        f"--reference {ref_path} "
        f"--celltype-col {celltype_col} "
        f"--data-dir {data_dir}"
        f"{gpu_flag}"
        f"{retrain_flag}"
        f"{no_rank_flag}"
        f"{sample_ids_flag}"
    )

    lines += [
        "",
        f"# CLASSIFIER — all samples under {data_dir}",
        f"# Trains once on reference, predicts on every *_reseg h5ad found at runtime",
        "",
        f"mkdir -p {log_dir}",
        "",
        f"cd {pipeline_root}",
        "",
        "echo '============================================'",
        f"echo '  CLASSIFIER — all samples'",
        f"echo '  Data dir: {data_dir}'",
        "echo \"  Job: $SLURM_JOB_ID  Node: $(hostname)\"",
        "echo \"  Start: $(date)\"",
        "echo '============================================'",
        "",
    ]

    lines += _container_block(site, container, py_args,
                              gpu=slurm.get("gpu", False), extra_binds=bind_paths)

    return "\n".join(lines)


def generate_multi_qc_script(
    cfg: dict,
    config_path: str,
    samples: list,
) -> str:
    """Generate ONE SLURM script that runs multi-sample QC on all samples in a slide.

    Passes --multi-sample --sample-ids id1 id2 ... --sample-dirs dir1 dir2 ...
    to run_qc.py, which concatenates h5ads across samples before computing QC.
    """
    method_cfg = get_method_config(cfg, "cellspa_qc")
    slurm = method_cfg["slurm"]
    container = get_container_path(cfg)
    output_base = get_output_base_override(cfg)
    pipeline_root = str(Path(config_path).resolve().parent.parent)
    site = load_site_config(pipeline_root)

    # All samples share the same slide_dir
    slide_dir = str(samples[0].slide_dir)
    slide_name = samples[0].slide_name

    log_dir = Path(pipeline_root) / "logs" / slide_name
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"seg_qc_combined_{slide_name}"
    python_script = METHOD_SCRIPTS["cellspa_qc"]

    # QC is CPU-only, so the GPU branch in _sbatch_directives never fires here.
    lines = _sbatch_directives(
        job_name, log_dir, slurm, site,
        defaults={"time": "0-12:00:00", "cpus_per_task": 4, "mem": "100G"},
    )

    sample_ids_str   = " ".join(s.sample_id  for s in samples)
    sample_dirs_str  = " ".join(str(s.sample_dir) for s in samples)

    ref_path = cfg.get("data", {}).get("reference_path", "")
    ref_arg  = f" --reference-path {ref_path}" if ref_path else ""

    bind_paths = set()
    bind_paths.add(slide_dir)
    bind_paths.add(str(log_dir))
    bind_paths.add(str(Path(config_path).resolve().parent))
    bind_paths.add(pipeline_root)
    for s in samples:
        bind_paths.add(str(s.sample_dir))
    if ref_path:
        bind_paths.add(str(Path(ref_path).parent))
    if output_base:
        bind_paths.add(output_base)

    python_bin = f"{site['python_bin']} -u"

    py_args = (
        f"    {python_bin} {python_script} "
        f"--config {config_path} "
        f"--slide-dir {slide_dir} "
        f"--multi-sample "
        f"--sample-ids {sample_ids_str} "
        f"--sample-dirs {sample_dirs_str}"
        f"{ref_arg}"
    )

    lines += [
        "",
        f"# MULTI-SAMPLE QC — {slide_name}",
        f"# Samples: {sample_ids_str}",
        "",
        f"mkdir -p {log_dir}",
        "",
        f"cd {pipeline_root}",
        "",
        "echo '============================================'",
        f"echo '  MULTI-SAMPLE QC — {slide_name}'",
        f"echo '  Samples: {sample_ids_str}'",
        "echo \"  Job: $SLURM_JOB_ID  Node: $(hostname)\"",
        "echo \"  Start: $(date)\"",
        "echo '============================================'",
        "",
    ]

    lines += _container_block(site, container, py_args, extra_binds=bind_paths)

    return "\n".join(lines)


