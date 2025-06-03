
# Uninstall problematic JAX/JAXlib
!pip uninstall jax jaxlib -y

# Install compatible JAX/JAXlib for CUDA 12
!pip install jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install jax==0.4.23

# Reinstall/Upgrade core libraries
!pip install diffusers transformers accelerate torch torchvision Pillow gradio --upgrade
