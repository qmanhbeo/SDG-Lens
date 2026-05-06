from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from pipeline_utils import MANUSCRIPT_DIR, ensure_base_dirs, rel_path, write_status


TEX_NAME = "sdg_lens_manuscript.tex"
REQUIRED_VIZ = [
    MANUSCRIPT_DIR / "visualization" / "tables" / "evaluation_summary_table.tex",
    MANUSCRIPT_DIR / "visualization" / "charts" / "model_comparison_micro_f1.png",
    MANUSCRIPT_DIR / "visualization" / "charts" / "model_comparison_macro_f1.png",
]
LATEX_TEMP_SUFFIXES = {
    ".aux", ".log", ".out", ".toc", ".fls",
    ".fdb_latexmk", ".nav", ".snm", ".bbl", ".blg",
    ".synctex.gz", ".ind", ".idx", ".loe", ".lof",
    ".lot", ".nlo", ".nls", ".pla", ".ps",
    ".tcp", ".tpt", ".trc",
}
LOG_FILES = {"compile_stdout.log", "compile_stderr.log"}


def cleanup_latex_artifacts(manuscript_dir: Path) -> list[str]:
    deleted: list[str] = []
    keep_dirs = {"visualization"}
    for path in manuscript_dir.iterdir():
        if path.is_dir():
            continue
        if path.suffix in LATEX_TEMP_SUFFIXES or path.name in LOG_FILES:
            path.unlink()
            deleted.append(rel_path(path))
    return deleted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile the SDG Lens LaTeX manuscript.")
    parser.add_argument("--tex", default=TEX_NAME)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_base_dirs()
    tex_path = MANUSCRIPT_DIR / args.tex
    if not tex_path.exists():
        raise FileNotFoundError(f"Missing manuscript source: {tex_path}")
    missing = [rel_path(p) for p in REQUIRED_VIZ if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required visualization assets:\n  " + "\n  ".join(missing) +
            "\nRun: python main.py visualize"
        )
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex was not found on PATH; cannot compile manuscript.")
    if args.dry_run:
        print(f"[compile] would compile {rel_path(tex_path)} with {pdflatex}")
        return 0

    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    stdout_log = MANUSCRIPT_DIR / "compile_stdout.log"
    stderr_log = MANUSCRIPT_DIR / "compile_stderr.log"
    failed = False
    for pass_idx in (1, 2):
        print(f"[compile] pdflatex pass {pass_idx}")
        result = subprocess.run(command, cwd=MANUSCRIPT_DIR, text=True, capture_output=True)
        stdout_log.write_text(result.stdout, encoding="utf-8")
        stderr_log.write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            failed = True
            tail = "\n".join(result.stdout.splitlines()[-20:])
            raise RuntimeError(
                f"LaTeX compilation failed on pass {pass_idx}. "
                f"See {rel_path(stdout_log)} and {rel_path(stderr_log)}.\n{tail}"
            )

    pdf_path = MANUSCRIPT_DIR / tex_path.with_suffix(".pdf").name
    if not pdf_path.exists():
        raise FileNotFoundError(f"LaTeX completed but did not produce PDF: {pdf_path}")
    write_status("compile_manuscript", "completed", "latex", pdf=rel_path(pdf_path), log=rel_path(stdout_log))
    print(f"[compile] wrote {rel_path(pdf_path)}")

    deleted = cleanup_latex_artifacts(MANUSCRIPT_DIR)
    for path in deleted:
        print(f"[compile] cleaned {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())