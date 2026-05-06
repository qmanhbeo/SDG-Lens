from __future__ import annotations

import argparse
import shutil
import subprocess

from pipeline_utils import MANUSCRIPT_DIR, ensure_base_dirs, rel_path, write_status


TEX_NAME = "sdg_lens_manuscript.tex"


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
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex was not found on PATH; cannot compile manuscript.")
    if args.dry_run:
        print(f"[compile] would compile {rel_path(tex_path)} with {pdflatex}")
        return 0

    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    stdout_log = MANUSCRIPT_DIR / "compile_stdout.log"
    stderr_log = MANUSCRIPT_DIR / "compile_stderr.log"
    for pass_idx in (1, 2):
        print(f"[compile] pdflatex pass {pass_idx}")
        result = subprocess.run(command, cwd=MANUSCRIPT_DIR, text=True, capture_output=True)
        stdout_log.write_text(result.stdout, encoding="utf-8")
        stderr_log.write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
