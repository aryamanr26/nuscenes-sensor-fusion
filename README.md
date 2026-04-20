# nuScenes Colorization Pipeline

Docker-first project scaffold for building a crisp, colorized point cloud pipeline on nuScenes.

## Project Layout

```text
nuscenes-colorization/
  Dockerfile
  requirements.txt
  config/scenes.yaml
  scripts/smoke_test.py
  src/pipeline.py
```

## Candidate Scenes

Configured in `config/scenes.yaml`:

- `0039`, `0054`, `0061`, `0066`, `0104`, `0108`, `0122`, `0176`, `0180`, `0193`

`selected_scene_ids` must contain at most 5 scenes.

## Build Docker Image

Build from `/home/aryamanr/nuscenes-colorization`:

```bash
docker build -t nuscenes-colorization \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER_NAME=$(whoami) \
  .
```

Why these args matter:

- container user is created with your host UID/GID
- mounted files remain owned by your host user
- default runtime user is non-root with passwordless sudo

## Run Container (GPU)

```bash
docker run --gpus all -it --rm \
  -v /home/aryamanr/nuscenes-colorization:/workspace/nuscenes-colorization \
  -v /mnt/workspace/datasets/nuscenes:/data/nuscenes \
  nuscenes-colorization
```

Inside container:

```bash
cd /workspace/nuscenes-colorization
python scripts/smoke_test.py \
  --dataroot /data/nuscenes \
  --version v1.0-trainval \
  --scenes-config config/scenes.yaml \
  --output-dir outputs/smoke
```

## Smoke Test Output

The smoke test verifies:

- selected scenes resolve correctly from metadata
- first selected scene is loadable
- LiDAR points project into `CAM_FRONT`
- a colored `.ply` cloud is written to `outputs/smoke/`

This gives a working baseline before implementing full multi-camera fusion and artifact mitigation.
