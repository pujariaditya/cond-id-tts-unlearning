#!/usr/bin/env bash
#
# download_vctk.sh — fetch the official VCTK-Corpus-0.92 into a local cache.
#
# Source : Edinburgh DataShare — "CSTR VCTK Corpus" (version 0.92)
#          Landing page : https://datashare.ed.ac.uk/handle/10283/3443
#          Dataset DOI  : 10.7488/ds/2645
# License : Creative Commons Attribution 4.0 International (CC-BY-4.0).
#           You must give appropriate credit when you use this corpus.
# Size    : ~10 GB compressed (zip), ~13 GB unpacked. Make sure you have the
#           disk space and a stable connection before running.
#
# This script is IDEMPOTENT: if the corpus already appears unpacked in the
# cache directory it skips the download and the unzip. Set VCTK_RAW_DIR to
# override the cache location.
#
# Usage:
#   ./scripts/download_vctk.sh
#   VCTK_RAW_DIR=/data/vctk ./scripts/download_vctk.sh
#
set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Cache directory: env override, else <repo>/data/_raw
RAW_DIR="${VCTK_RAW_DIR:-${REPO_ROOT}/data/_raw}"

# Direct download of the version-0.92 zip from DataShare.
ZIP_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
ZIP_NAME="VCTK-Corpus-0.92.zip"
ZIP_PATH="${RAW_DIR}/${ZIP_NAME}"

# A directory that exists only once the corpus is unpacked. VCTK-0.92 ships
# the per-speaker audio under wav48_silence_trimmed/.
MARKER_DIR="${RAW_DIR}/wav48_silence_trimmed"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
log() { printf '[download_vctk] %s\n' "$*"; }

# --------------------------------------------------------------------------
# Banner / license notice
# --------------------------------------------------------------------------
log "==============================================================="
log " CSTR VCTK Corpus 0.92"
log "   landing : https://datashare.ed.ac.uk/handle/10283/3443"
log "   DOI     : 10.7488/ds/2645"
log "   license : CC-BY-4.0 (attribution required)"
log "   size    : ~10 GB download, ~13 GB unpacked"
log "   cache   : ${RAW_DIR}"
log "==============================================================="

# --------------------------------------------------------------------------
# Idempotency: already unpacked?
# --------------------------------------------------------------------------
if [ -d "${MARKER_DIR}" ] && [ -n "$(ls -A "${MARKER_DIR}" 2>/dev/null || true)" ]; then
    log "Corpus already present at ${MARKER_DIR} — nothing to do."
    exit 0
fi

mkdir -p "${RAW_DIR}"

# --------------------------------------------------------------------------
# Download (resumable) unless the zip is already here.
# --------------------------------------------------------------------------
if [ -f "${ZIP_PATH}" ]; then
    log "Found existing archive ${ZIP_PATH} — skipping download."
else
    log "Downloading ${ZIP_NAME} (~10 GB). This may take a while ..."
    if command -v curl >/dev/null 2>&1; then
        # -C - resumes a partial download; -L follows redirects.
        curl -L -C - --fail -o "${ZIP_PATH}.part" "${ZIP_URL}"
        mv "${ZIP_PATH}.part" "${ZIP_PATH}"
    elif command -v wget >/dev/null 2>&1; then
        # -c continues a partial download.
        wget -c -O "${ZIP_PATH}.part" "${ZIP_URL}"
        mv "${ZIP_PATH}.part" "${ZIP_PATH}"
    else
        log "ERROR: neither curl nor wget is available." >&2
        exit 1
    fi
    log "Download complete: ${ZIP_PATH}"
fi

# --------------------------------------------------------------------------
# Unzip
# --------------------------------------------------------------------------
if ! command -v unzip >/dev/null 2>&1; then
    log "ERROR: 'unzip' is required to unpack the corpus." >&2
    exit 1
fi

log "Unpacking ${ZIP_NAME} into ${RAW_DIR} ..."
unzip -q -o "${ZIP_PATH}" -d "${RAW_DIR}"

if [ -d "${MARKER_DIR}" ] && [ -n "$(ls -A "${MARKER_DIR}" 2>/dev/null || true)" ]; then
    log "Unpacked successfully. Audio under ${MARKER_DIR}"
    log "You may delete the archive to reclaim space: rm '${ZIP_PATH}'"
else
    log "ERROR: expected ${MARKER_DIR} after unzip but it is missing/empty." >&2
    log "       Inspect ${RAW_DIR} — the archive layout may have changed." >&2
    exit 1
fi

log "Done. Next: python scripts/prepare_vctk.py --raw-dir '${RAW_DIR}'"
