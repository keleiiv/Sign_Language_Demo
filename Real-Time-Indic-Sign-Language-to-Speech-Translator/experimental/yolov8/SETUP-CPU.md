# Isolated CPU-only YOLOv8 environment

This environment is for the experimental YOLOv8 work only. It uses Python 3.11
and does not alter the project virtual environment or application dependencies.
Run commands from the repository's `Real-Time-Indic-Sign-Language-to-Speech-Translator`
directory.

## Create the environment

```powershell
py -3.11 -m venv experimental\yolov8\.venv
experimental\yolov8\.venv\Scripts\python.exe -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu
experimental\yolov8\.venv\Scripts\python.exe -m pip install --no-cache-dir ultralytics==8.4.183
```

The first install command follows the official PyTorch CPU wheel index. The
second installs Ultralytics from PyPI after PyTorch and torchvision. The pinned
versions are also listed in `requirements-cpu.txt`.

## Verify imports and architecture

```powershell
$env:YOLO_CONFIG_DIR = (Resolve-Path 'experimental\yolov8').Path + '\.ultralytics'
experimental\yolov8\.venv\Scripts\python.exe -c "import torch, torchvision, ultralytics; from ultralytics import YOLO; model=YOLO('yolov8n.yaml'); print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('ultralytics',ultralytics.__version__); print('cuda_available',torch.cuda.is_available()); print('tensor',torch.ones((2, 2), device='cpu')); print('model_task',model.task); print('parameters',sum(p.numel() for p in model.model.parameters()))"
```

Using `yolov8n.yaml` constructs the small detection architecture from its
configuration; it does not fetch pretrained weights or a dataset. Training
must use `experimental\yolov8\prepared\data.yaml`, not the source dataset
configuration.

This setup file provisions dependencies only. It does not start training,
download model weights, or integrate the experiment into the application.
