# Copyright 2024 ByteDance and/or its affiliates.
#
# Copyright (2024) X-Dyna Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

# Install Anaconda or Miniconda
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install face_alignment
pip install pyvideoreader
pip install imageio[ffmpeg] 
pip install moviepy
pip install diffusers==0.24.0 
pip install joblib
pip install scikit-image
pip install visdom
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom tqdm scipy plotly scikit-learn imageio[ffmpeg] gradio trimesh huggingface_hub
pip uninstall numpy -y
pip install numpy==1.26.3
pip uninstall xformers -y
pip uninstall torch torchvision torchaudio -y
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.20
pip install huggingface_hub==0.25.2
