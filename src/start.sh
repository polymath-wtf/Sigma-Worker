#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

# ---------------------------------------------------------------------------
# Cached model support (RunPod model caching via HuggingFace)
# ---------------------------------------------------------------------------
# When a cached model is configured on the RunPod endpoint, the HF repo is
# available at /runpod-volume/huggingface-cache/hub/models--<org>--<repo>/snapshots/<hash>/
# We symlink every model file found there into the matching ComfyUI model directory.
#
# Set CACHED_MODEL_REPO to the HuggingFace repo id used in the RunPod endpoint
# "Model" field, e.g. "Seryoger/runpod-endpoint-cache".
# If not set, we auto-detect any repo present in the cache directory.

CACHE_DIR="/runpod-volume/huggingface-cache/hub"
COMFYUI_MODELS="/comfyui/models"

link_cached_models() {
    local snapshot_dir="$1"
    local models_root="${snapshot_dir}/models"

    if [ ! -d "$models_root" ]; then
        echo "worker-comfyui: [cache] No 'models/' directory in snapshot, skipping: ${snapshot_dir}"
        return
    fi

    echo "worker-comfyui: [cache] Linking cached models from ${models_root}"

    # Walk every file under the snapshot's models/ tree
    find "$models_root" -type f | while read -r src_file; do
        # Relative path inside models/, e.g. "unet/flux-2-klein-9b-fp8.safetensors"
        rel_path="${src_file#${models_root}/}"
        dest_file="${COMFYUI_MODELS}/${rel_path}"
        dest_dir="$(dirname "$dest_file")"

        # Skip placeholder .txt files (used to keep empty dirs in git)
        case "$src_file" in *.txt) continue ;; esac

        # Create target directory if it doesn't exist
        mkdir -p "$dest_dir"

        # Only create symlink if target doesn't already exist (baked-in model takes priority)
        if [ ! -e "$dest_file" ]; then
            ln -sf "$src_file" "$dest_file"
            echo "worker-comfyui: [cache]   linked: ${rel_path}"
        else
            echo "worker-comfyui: [cache]   skipped (already exists): ${rel_path}"
        fi
    done
}

if [ -d "$CACHE_DIR" ]; then
    echo "worker-comfyui: [cache] HuggingFace cache directory found at ${CACHE_DIR}"

    if [ -n "$CACHED_MODEL_REPO" ]; then
        # Use the explicitly configured repo
        cache_name="models--$(echo "$CACHED_MODEL_REPO" | sed 's|/|--|g')"
        repo_dir="${CACHE_DIR}/${cache_name}"
    else
        # Auto-detect: pick the first repo directory in the cache
        repo_dir=""
        for d in "${CACHE_DIR}"/models--*; do
            if [ -d "$d" ]; then
                repo_dir="$d"
                break
            fi
        done
    fi

    if [ -n "$repo_dir" ] && [ -d "$repo_dir" ]; then
        echo "worker-comfyui: [cache] Using cached repo: ${repo_dir}"

        # Resolve the latest snapshot via refs/main, or fall back to first snapshot
        snapshot_hash=""
        if [ -f "${repo_dir}/refs/main" ]; then
            snapshot_hash="$(cat "${repo_dir}/refs/main" | tr -d '[:space:]')"
        fi

        snapshot_dir=""
        if [ -n "$snapshot_hash" ] && [ -d "${repo_dir}/snapshots/${snapshot_hash}" ]; then
            snapshot_dir="${repo_dir}/snapshots/${snapshot_hash}"
        else
            # Fall back to first available snapshot
            for s in "${repo_dir}/snapshots"/*/; do
                if [ -d "$s" ]; then
                    snapshot_dir="${s%/}"
                    break
                fi
            done
        fi

        if [ -n "$snapshot_dir" ]; then
            echo "worker-comfyui: [cache] Snapshot directory: ${snapshot_dir}"
            link_cached_models "$snapshot_dir"
            echo "worker-comfyui: [cache] Model linking complete"
        else
            echo "worker-comfyui: [cache] WARNING: No snapshots found in ${repo_dir}"
        fi
    else
        echo "worker-comfyui: [cache] No cached model repo found in ${CACHE_DIR}"
    fi
else
    echo "worker-comfyui: [cache] No HuggingFace cache directory at ${CACHE_DIR} (not using cached models)"
fi

# ---------------------------------------------------------------------------
# GPU detection – skip ComfyUI when no GPU is available (RunPod test phase)
# ---------------------------------------------------------------------------
# RunPod's GitHub-integration build pipeline runs a mandatory "Testing" phase
# after pushing the image.  The test container has NO GPU, so ComfyUI hangs
# forever trying to initialise CUDA.  We detect this and start only the
# RunPod handler, which immediately reports "ready" and lets the test pass.
if ! nvidia-smi > /dev/null 2>&1; then
    echo "worker-comfyui: WARNING – No GPU detected (RunPod test environment?)"
    echo "worker-comfyui: Skipping ComfyUI, starting handler-only so the test passes."
    python -u /handler.py
    exit $?
fi

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi