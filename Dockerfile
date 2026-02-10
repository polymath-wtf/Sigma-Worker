# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi
    
# Install Triton + SageAttention
RUN uv pip install --upgrade "triton==3.5.1" \
    && wget -q -O /tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl \
      "https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl" \
    && uv pip install /tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl \
    && rm -f /tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
    
# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

RUN comfy-node-install \
    comfyui-videohelpersuite \
    comfyui-frame-interpolation \
    comfymath \
    rgthree-comfy \
    comfyui-gguf \
    comfyui-wanvideowrapper \
    comfyui-kjnodes \
    comfyui-multigpu \
    comfyui-easy-use \
    was-node-suite-comfyui \
    comfyui-custom-scripts \
    comfyui_controlnet_aux \
    comfyui_layerstyle \
    comfyui_essentials \
    cg-use-everywhere \
#    comfyui-tripleksampler \ not in https://registry.comfy.org/ru 
    comfyui-mediamixer \
    comfyui-wanmoeksampler \
    comfyui_ultimatesdupscale \
    comfyui_fill-nodes \
    comfyui-ic-light \
    comfyui-art-venture \
    efficiency-nodes-comfyui \
    comfyUI_fizzNodes \
    comfyui-painterfluxImageedit
    
# Install KJnodes from GitHub (manual comit)
RUN git clone https://github.com/polymath-wtf/ComfyUI-Polymath-Vibenodes.git /comfyui/custom_nodes/ComfyUI-Polymath-Vibenodes

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG CIVITAI_ACCESS_TOKEN

# Set default model type if none is provided
ARG MODEL_TYPE=flux2-klein-9b-fp8

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/text_encoders models/diffusion_models models/model_patches

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN if [ "$MODEL_TYPE" = "Wan_i2v_default" ]; then \
      wget -q -O models/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors && \
      wget -q -O models/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors && \
      wget -q -O models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
      wget -q -O models/clip_vision/clip_vision_h.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors && \
      wget -q -O models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors && \
      wget -q -O models/loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors && \
      wget -q -O models/loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors; \
    fi



RUN if [ "$MODEL_TYPE" = "Wan_i2v_dasiwa" ]; then \
      wget -q -O models/unet/DasiwaWAN22I2V14BLightspeed_synthseductionHighV9.safetensors https://civitai.com/api/download/models/2555640?token=--header="Authorization: Bearer ${CIVITAI_ACCESS_TOKEN}" && \
      wget -q -O models/unet/DasiwaWAN22I2V14BLightspeed_synthseductionLowV9.safetensors https://civitai.com/api/download/models/2555652?token=--header="Authorization: Bearer ${CIVITAI_ACCESS_TOKEN}" && \
      wget -q -O models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
      wget -q -O models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors && \
#      wget -q -O models/clip_vision/clip_vision_h.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors && \
#      wget -q -O models/loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors && \
#      wget -q -O models/loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors && \
      wget -q -O models/loras/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors && \
      wget -q -O models/loras/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors; \
    fi
    
RUN if [ "$MODEL_TYPE" = "flux1-krea" ]; then \
      wget -q -O models/unet/flux1-krea-dev_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/FLUX.1-Krea-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q -O models/vae/ae.safetensors https://huggingface.co/Seryoger/Parique_v1/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux2-klein-9b-fp8" ]; then \
      wget -q --header="Authorization: Bearer $HUGGINGFACE_ACCESS_TOKEN" -O /comfyui/models/unet/flux-2-klein-9b-fp8.safetensors https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors && \
      wget -q -O models/clip/qwen_3_8b_fp8mixed.safetensors https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors && \
      wget -q -O models/vae/flux2-vae.safetensors https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors; \

RUN if [ "$MODEL_TYPE" = "z-image-turbo" ]; then \
      wget -q -O models/text_encoders/qwen_3_4b.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors && \
      wget -q -O models/diffusion_models/z_image_turbo_bf16.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors && \
      wget -q -O models/vae/ae.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors && \
      wget -q -O models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors; \
    fi
    
RUN if [ "$MODEL_TYPE" = "fast" ]; then \
      wget -q -O models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
