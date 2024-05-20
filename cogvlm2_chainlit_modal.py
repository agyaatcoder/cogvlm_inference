import modal
import subprocess

GPU_CONFIG = modal.gpu.A100(count=1, memory = 80)

#TODOS -Add model to model store and use it for inferencing
#volume = modal.Volume.from_name("hf-model-store", create_if_missing=True)

MINUTES = 60 

cogvlm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/agyaatcoder/CogVLM2"
    ).workdir("CogVLM2/basic_demo")
    .run_commands("pip install -r requirements.txt")
)

stub = modal.Stub("app")


@stub.function(
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
    image=cogvlm_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu = GPU_CONFIG
)
@modal.web_server(8000, startup_timeout=20 * MINUTES)
def run():
    target = 'web_demo.py'
    cmd = f"chainlit run {target} --port 8000"
    subprocess.Popen(cmd, shell=True)
