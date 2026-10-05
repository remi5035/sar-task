#!/bin/bash
# Exécute sar_diagnostic.ipynb sur place (les sorties restent dans le notebook, lisible sur GitHub)
# et exporte sar_diagnostic_rapport.pdf (sorties seules : texte, tables, figures). Windows, Git Bash.
# Usage : bash build_pdf.sh [--sync RUN...]   (--sync copie les runs depuis WSL ~/sar-task/analysis/data)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONIOENCODING=utf-8 PYDEVD_DISABLE_FILE_VALIDATION=1

if [[ "${1:-}" == "--sync" ]]; then
  shift
  for run in "$@"; do
    echo "sync $run"
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -c \
      "mkdir -p '/mnt/d/projets IA/SWARM/sar-task/analysis/data/$run' && cp -ru ~/sar-task/analysis/data/$run/. '/mnt/d/projets IA/SWARM/sar-task/analysis/data/$run/'"
    rm -f "../data/$run/_traj_features.pkl"   # le cache de features doit suivre les épisodes
  done
fi

mkdir -p pdf
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 sar_diagnostic.ipynb
python -m jupyter nbconvert --to html --no-input --output-dir pdf --output sar_diagnostic_rapport sar_diagnostic.ipynb

python - <<'EOF'
from pathlib import Path
css = """<style>
@page { size: A4; margin: 12mm 10mm; }
body { font-size: 11px; }
.jp-Notebook { padding: 0 !important; }
.jp-Cell { break-inside: auto; }
.jp-OutputArea-output img { max-width: 100% !important; height: auto !important; break-inside: avoid; }
.jp-RenderedHTMLCommon table { font-size: 9.5px; }
pre, .highlight pre { white-space: pre-wrap !important; word-break: break-word; }
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
h1, h2 { break-before: auto; break-after: avoid; }
</style>"""
for name in ("sar_diagnostic_rapport",):
    p = Path("pdf") / f"{name}.html"
    t = p.read_text(encoding="utf-8")
    p.write_text(t.replace("</head>", css + "</head>", 1), encoding="utf-8")
EOF

EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
WIN_PDF="$(cygpath -w "$HERE/pdf")"
for name in sar_diagnostic_rapport; do
  url="file:///$(cygpath -m "$HERE/pdf/$name.html" | sed 's/ /%20/g')"
  rm -f "pdf/$name.pdf"
  "$EDGE" --headless=new --disable-gpu --no-pdf-header-footer --user-data-dir='D:\edge-tmp' \
    --virtual-time-budget=30000 "--print-to-pdf=$WIN_PDF\\$name.pdf" "$url" 2>/dev/null || true
  mv -f "pdf/$name.pdf" "$name.pdf" && ls -la "$name.pdf"
done
