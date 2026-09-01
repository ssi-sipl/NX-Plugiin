# Environment Setup Guide: OpenVINO → Triton Migration (Nx Network Optix Plugin)

This document captures the exact, confirmed-working sequence for setting up a development environment for the YOLOv8s + NVIDIA Triton Inference Server integration with the Network Optix analytics plugin. Every command below was actually run and verified during development — steps marked **[NOT CONFIRMED]** were assumed pre-existing and should be checked on the source machine before replicating.

---

## Target Hardware / OS Baseline

Confirmed on development machine:

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS x86_64 |
| CPU | Intel i9-14900F |
| GPU | NVIDIA GeForce RTX 5060 Ti (Blackwell, compute capability 12.0) |
| VRAM | 15,849 MiB |

**[NOT CONFIRMED — check before replicating]**
```bash
nvidia-smi --query-gpu=driver_version --format=csv
docker --version
```
Record these from the working machine. The `tritonserver:25.12-py3` container requires CUDA 13.0 (confirmed via `libnvinfer-bin 10.14.1.48-1+cuda13.0` inside the image), which needs a correspondingly recent host driver — match this on the target machine rather than assuming a specific number.

---

## 1. Base OS Setup

```bash
sudo apt update && sudo apt upgrade -y
lsb_release -a   # expect Ubuntu 24.04.4 LTS
```

## 2. NVIDIA Driver + Docker + NVIDIA Container Toolkit

**[NOT CONFIRMED in this session — assumed pre-existing]**

```bash
sudo apt install -y nvidia-driver-<matching-version-from-source-machine>
sudo apt install -y docker.io
sudo systemctl enable --now docker

distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU passthrough works:
```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## 3. Python Environment for Model Export

Confirmed working: venv + ultralytics, ONNX 1.22.0, onnxslim 0.1.95.

```bash
mkdir -p ~/Projects/Ultarlytics && cd ~/Projects/Ultarlytics
python3 -m venv venv
source venv/bin/activate
pip install ultralytics onnx
```

## 4. Export YOLOv8s to Dynamic ONNX

```bash
yolo export model=yolov8s.pt format=onnx opset=17 imgsz=640 dynamic=True simplify=True
```

**Note:** `dynamic=True` is required — without it, the ONNX graph has a static batch dimension and `trtexec`'s `--minShapes/--optShapes/--maxShapes` flags will fail with "Static model does not take explicit shapes."

Verify tensor shapes:
```bash
python3 -c "
import onnx
m = onnx.load('yolov8s.onnx')
print('inputs:', [(i.name, [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print('outputs:', [(o.name, [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
"
```
Expected: `images: ['batch', 3, 'height', 'width']`, `output0: ['batch', 84, 'anchors']`

## 5. Pull the Pinned Triton Container

```bash
docker pull nvcr.io/nvidia/tritonserver:25.12-py3
```

**Version pin rationale:** `26.07-py3` ships TensorRT 11.1, which removed the classic `--fp16`/`--int8`/`--best` precision flags entirely (strong typing required instead, via NVIDIA ModelOpt AutoCast). `25.12-py3` still ships **TensorRT 10.14.1** with `--fp16` fully supported — use this pin unless the plugin is later updated to support strongly-typed ONNX.

Confirmed inside this image: Triton Server **2.64.0**, TensorRT **10.14.1**, CUDA 13.0.

## 6. Build the TensorRT Engine

```bash
docker run --gpus all -it --rm -v ~/Projects/Ultarlytics:/workspace nvcr.io/nvidia/tritonserver:25.12-py3 bash
```

Inside the container — `trtexec` is not on `PATH` by default in this image:
```bash
export PATH=$PATH:/usr/src/tensorrt/bin

trtexec --onnx=/workspace/yolov8s.onnx \
  --saveEngine=/workspace/model.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:16x3x640x640 \
  --maxShapes=images:32x3x640x640
```

Confirm `&&&& PASSED` in the output before proceeding. Exit the container:
```bash
exit
```

## 7. Model Repository

```bash
mkdir -p ~/triton_models/yolov8s_person/1
cp ~/Projects/Ultarlytics/model.plan ~/triton_models/yolov8s_person/1/model.plan
```

Create `~/triton_models/yolov8s_person/config.pbtxt`:
```protobuf
name: "yolov8s_person"
platform: "tensorrt_plan"
max_batch_size: 32

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 15000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

## 8. Launch Triton Server

**Check port availability first.** On the original development machine, Nx mediaserver already held port 8001 for inter-server communication, requiring a remap. On a fresh machine without Nx installed, default ports may be free:

```bash
sudo lsof -i :8000
sudo lsof -i :8001
sudo lsof -i :8002
```

If clear, use default ports. If occupied (e.g. by Nx mediaserver), remap as done on the original machine:

```bash
docker run --gpus all -d --name triton_server \
  -p 9000:8000 -p 9001:8001 -p 9002:8002 \
  -v ~/triton_models:/models \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models
```

## 9. Verify Server

```bash
docker logs -f triton_server
curl localhost:9000/v2/health/ready
curl localhost:9000/v2/models/yolov8s_person
```

Expected model metadata response:
```json
{"name":"yolov8s_person","versions":["1"],"platform":"tensorrt_plan","inputs":[{"name":"images","datatype":"FP32","shape":[-1,3,640,640]}],"outputs":[{"name":"output0","datatype":"FP32","shape":[-1,84,8400]}]}
```

**Locked port mapping (if remapped):**
- HTTP: `localhost:9000`
- gRPC: `localhost:9001` ← required by the plugin's Triton client connection string
- Metrics: `localhost:9002`

## 10. Extract Triton Client SDK

Required for building the plugin against the Triton gRPC client library.

```bash
mkdir -p ~/triton_client
docker run --rm -v ~/triton_client:/out nvcr.io/nvidia/tritonserver:25.12-py3-sdk \
  bash -c "cp -r /workspace/install/* /out/"
```

Verify extraction:
```bash
ls ~/triton_client/include | grep -i grpc   # expect grpc_client.h
ls ~/triton_client/lib | grep -i grpc       # expect libgrpcclient.so
```

## 11. Network Optix Metadata SDK + Plugin Source

**[Environment-specific — not run fresh in this session]**

Obtain the same Metadata SDK version used in development (`nxwitness-metadata_sdk-6.0.6.41837-universal` or current equivalent) from the Nx partner portal and unpack it.

Place the plugin source — `device_agent.h`, `device_agent.cpp`, `CMakeLists.txt` (Triton-integrated versions) — under:
```
<metadata_sdk>/samples/stub_analytics_plugin/
```

## 12. Configure and Build the Plugin

```bash
cd <metadata_sdk>/samples/stub_analytics_plugin
cmake -B build \
  -DmetadataSdkDir=<path_to_metadata_sdk> \
  -DtritonClientDir=$HOME/triton_client \
  .
cmake --build build -j$(nproc)
```

Output: `build/libstub_analytics_plugin.so`

---

## Post-Setup Validation

Before deploying to production, rerun the throughput sweep on the new machine — hardware and driver differences between machines can silently change performance characteristics even when the build succeeds identically:

```bash
docker run --rm --net=host nvcr.io/nvidia/tritonserver:25.12-py3-sdk \
  perf_analyzer -m yolov8s_person \
  -i grpc \
  -u localhost:9001 \
  --shape images:3,640,640 \
  --concurrency-range 8:32:8 \
  --measurement-mode count_windows \
  --measurement-request-count 200
```

---

## Known Issues Encountered During Development

| Issue | Cause | Fix |
|---|---|---|
| `trtexec: unknown option --fp16` | Running TensorRT 11.x container (`26.07-py3`), which removed classic precision flags | Pin to `25.12-py3` (TensorRT 10.14.1) |
| `trtexec: command not found` | `trtexec` not on `PATH` in `tritonserver` image | `export PATH=$PATH:/usr/src/tensorrt/bin` |
| `Static model does not take explicit shapes` | ONNX exported without `dynamic=True` | Re-export with `dynamic=True` |
| `bind host port 0.0.0.0:8001: address already in use` | Nx mediaserver already listening on 8001 for inter-server comms | Remap Triton ports (9000/9001/9002 → 8000/8001/8002) |
| `needToDetectObjects()` hardcoded to `return true` | Bypassed the plugin's own adaptive-FPS throttling, causing every decoded frame to trigger inference | Restored the frame-period gate; set `m_detectionFrequencyFps` to a real target (5/sec) instead of the `1000` sentinel |
