"""Compile the SDG Lens manuscript after checking required generated assets.

LaTeX failures are easier to diagnose when the pipeline verifies inputs first,
captures compiler logs, and removes temporary files after a successful build.
This stage therefore behaves more like a reproducible build step than a bare
``pdflatex`` invocation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pipeline_utils import MANUSCRIPT_DIR, ensure_base_dirs, rel_path, write_status


TEX_NAME = "sdg_lens_manuscript.tex"
COVERSHEET_DOCX = "coversheet.docx"
COVERSHEET_PDF = "coversheet.pdf"
SUBMISSION_PDF = "sdg_lens_submission.pdf"

# These files are referenced by the manuscript. Failing early with a clear list
# is more useful than letting LaTeX stop on a missing graphic/table later.
REQUIRED_VIZ = [
    MANUSCRIPT_DIR / "visualization" / "tables" / "evaluation_summary_table.tex",
    MANUSCRIPT_DIR / "visualization" / "charts" / "model_comparison_micro_f1.png",
    MANUSCRIPT_DIR / "visualization" / "charts" / "model_comparison_macro_f1.png",
]

# Only known LaTeX byproducts are cleaned. The TeX source, PDF, and
# visualization subdirectory are intentionally left untouched.
LATEX_TEMP_SUFFIXES = {
    ".aux", ".log", ".out", ".toc", ".fls",
    ".fdb_latexmk", ".nav", ".snm", ".bbl", ".blg",
    ".synctex.gz", ".ind", ".idx", ".loe", ".lof",
    ".lot", ".nlo", ".nls", ".pla", ".ps",
    ".tcp", ".tpt", ".trc",
}
LOG_FILES = {"compile_stdout.log", "compile_stderr.log"}


def cleanup_latex_artifacts(manuscript_dir: Path) -> list[str]:
    """Remove temporary LaTeX files while reporting exactly what was deleted."""
    deleted: list[str] = []
    keep_dirs = {"visualization"}
    for path in manuscript_dir.iterdir():
        if path.is_dir():
            continue
        if path.suffix in LATEX_TEMP_SUFFIXES or path.name in LOG_FILES:
            path.unlink()
            deleted.append(rel_path(path))
    return deleted


def run_checked(command: list[str], cwd: Path, action: str, env: dict[str, str] | None = None) -> None:
    """Run a build command and surface concise diagnostics if it fails."""
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-20:])
        raise RuntimeError(f"{action} failed with exit code {result.returncode}.\n{tail}")


def build_submission_pdf(manuscript_dir: Path, manuscript_pdf: Path) -> Path:
    """Convert the coversheet and merge it with the compiled manuscript PDF."""
    coversheet_docx = manuscript_dir / COVERSHEET_DOCX
    coversheet_pdf = manuscript_dir / COVERSHEET_PDF
    submission_pdf = manuscript_dir / SUBMISSION_PDF

    if not coversheet_docx.exists():
        raise FileNotFoundError(f"Missing coversheet source: {coversheet_docx}")
    libreoffice = shutil.which("libreoffice")
    if libreoffice is None:
        raise RuntimeError("libreoffice was not found on PATH; cannot convert coversheet.docx to PDF.")
    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("pdfunite was not found on PATH; cannot merge coversheet and manuscript PDFs.")

    with tempfile.TemporaryDirectory(prefix="sdg-lens-libreoffice-") as tmp:
        tmp_path = Path(tmp)
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(mode=0o700)
        lo_env = os.environ.copy()
        lo_env.update({
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "XDG_RUNTIME_DIR": str(runtime_dir),
        })
        user_installation = tmp_path / "profile"
        convert_command = [
            libreoffice,
            "--headless",
            f"-env:UserInstallation=file://{user_installation}",
            "--convert-to",
            "pdf",
            COVERSHEET_DOCX,
        ]
        print(f"[compile] convert: {' '.join(convert_command)}")
        run_checked(convert_command, cwd=manuscript_dir, action="Coversheet conversion", env=lo_env)
    if not coversheet_pdf.exists():
        raise FileNotFoundError(f"LibreOffice completed but did not produce PDF: {coversheet_pdf}")

    merge_command = [pdfunite, COVERSHEET_PDF, manuscript_pdf.name, SUBMISSION_PDF]
    print(f"[compile] merge: {' '.join(merge_command)}")
    run_checked(merge_command, cwd=manuscript_dir, action="Submission PDF merge")
    if not submission_pdf.exists():
        raise FileNotFoundError(f"pdfunite completed but did not produce PDF: {submission_pdf}")
    return submission_pdf


def build_parser() -> argparse.ArgumentParser:
    """Build the direct-stage CLI used by the orchestrator."""
    parser = argparse.ArgumentParser(description="Compile the SDG Lens LaTeX manuscript.")
    parser.add_argument("--tex", default=TEX_NAME)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Validate inputs, run two LaTeX passes, and publish the resulting PDF."""
    args = build_parser().parse_args()
    ensure_base_dirs()
    tex_path = MANUSCRIPT_DIR / args.tex
    if not tex_path.exists():
        raise FileNotFoundError(f"Missing manuscript source: {tex_path}")

    # Check generated assets before invoking xelatex so the recovery action is
    # obvious: regenerate visualizations rather than debug a LaTeX include error.
    missing = [rel_path(p) for p in REQUIRED_VIZ if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required visualization assets:\n  " + "\n  ".join(missing) +
            "\nRun: python main.py visualize"
        )
    xelatex = shutil.which("xelatex")
    if xelatex is None:
        raise RuntimeError("xelatex was not found on PATH; cannot compile manuscript.")
    if args.dry_run:
        # Dry-run still validates the TeX file, required assets, and xelatex
        # availability, because those are the common compile blockers.
        print(f"[compile] would compile {rel_path(tex_path)} with {xelatex}")
        print(f"[compile] would convert {rel_path(MANUSCRIPT_DIR / COVERSHEET_DOCX)} to {COVERSHEET_PDF}")
        print(f"[compile] would merge {COVERSHEET_PDF} + {tex_path.with_suffix('.pdf').name} -> {SUBMISSION_PDF}")
        return 0

    command = [xelatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    stdout_log = MANUSCRIPT_DIR / "compile_stdout.log"
    stderr_log = MANUSCRIPT_DIR / "compile_stderr.log"
    failed = False
    for pass_idx in (1, 2):
        # Two passes are enough for the simple cross-reference/table layout used
        # here and avoid adding latexmk as an extra runtime dependency.
        print(f"[compile] pdflatex pass {pass_idx}")
        result = subprocess.run(command, cwd=MANUSCRIPT_DIR, text=True, capture_output=True)
        stdout_log.write_text(result.stdout, encoding="utf-8")
        stderr_log.write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            failed = True
            # Include only the tail in the exception; the complete logs remain
            # on disk for detailed inspection.
            tail = "\n".join(result.stdout.splitlines()[-20:])
            raise RuntimeError(
                f"LaTeX compilation failed on pass {pass_idx}. "
                f"See {rel_path(stdout_log)} and {rel_path(stderr_log)}.\n{tail}"
            )

    pdf_path = MANUSCRIPT_DIR / tex_path.with_suffix(".pdf").name
    if not pdf_path.exists():
        raise FileNotFoundError(f"LaTeX completed but did not produce PDF: {pdf_path}")

    print(f"[compile] wrote {rel_path(pdf_path)}")

    submission_pdf = build_submission_pdf(MANUSCRIPT_DIR, pdf_path)
    print(f"[compile] wrote {rel_path(submission_pdf)}")

    # Publish status after confirming both PDFs exist so downstream checks do
    # not mistake a failed submission merge for a completed build.
    write_status(
        "compile_manuscript",
        "completed",
        "latex",
        pdf=rel_path(pdf_path),
        submission_pdf=rel_path(submission_pdf),
        log=rel_path(stdout_log),
    )

    # Cleanup happens after status is written; if cleanup ever fails, the PDF and
    # logs still tell the important story of the compile.
    deleted = cleanup_latex_artifacts(MANUSCRIPT_DIR)
    for path in deleted:
        print(f"[compile] cleaned {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
