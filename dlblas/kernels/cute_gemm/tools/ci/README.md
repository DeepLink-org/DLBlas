# cute_gemm CI

CI runs on self-hosted GPU runners (h100, b300, h100/sm120) inside an Apptainer
SIF pulled from a Docker image on Docker Hub. Triggered on every push to `main`
and on PRs.

## Files

| File | Purpose |
|------|---------|
| `tools/ci/docker/Dockerfile` | image recipe (one Dockerfile, two variants via build args) |
| `tools/ci/docker/build.sh` | step flags `--image` / `--push` / `--sif` per variant: build the docker image, push to `<owner>/dl-kernels`, and/or build the runner's cached SIF |
| `.github/workflows/_test.yml` | reusable workflow with lint/changes/test jobs and the matrix; **the image tag pins live here** |
| `.github/workflows/ci.yml`, `ci-pr.yml` | thin shells that call `_test.yml` on push / PR |
| `.github/actions/gpu-test/action.yml` | composite action — pulls SIF, runs single-pass pytest |

## Image variants

| Variant | Docker image (latest) | Notes |
|---------|------------------------|-------|
| `cu129` | `<owner>/dl-kernels:cu12.9-DATE` | base cute-dsl |
| `cu132` | `<owner>/dl-kernels:cu13.2-DATE` | cute-dsl[cu13] |

CI reads the image namespace from the `DLKERNEL_IMAGE_REPO` repository variable
(required — set it to the lowercase Docker Hub namespace that holds the images,
e.g. `myuser/dl-kernels`). `build.sh` picks the namespace up from
`DOCKERHUB_USER` / `REGISTRY_REPO`, falling back to the (lowercased) owner of
the `origin` git remote — point both at the same namespace or CI will re-pull
(or fail) after you push.

The cu12.9 variant uses PyTorch 2.14 cu126 wheels because cu129 wheels are no
longer published. The `cu129` variant name and `cu12.9` image tag are retained
for the CUDA 12 runner configuration; they do not identify torch's wheel version.
The cu13.2 variant uses PyTorch 2.14 cu132 wheels plus the Dockerfile's CUDA 13 forward-
compatibility libcuda shim so it remains runnable on 575-series kernel drivers.

## Test matrix

`_test.yml` runs 5 jobs per push:

| GPU | Arch override | cu129 | cu132 |
|-----|----------------|-------|-------|
| h100 | (none, sm90) | ✓ | ✓ |
| b300 | (none, sm100) | — | ✓ |
| h100 | sm120 | ✓ | ✓ |

The cu126 torch wheels lack Blackwell support, so B300 runs only with cu132.
The h100/sm120 jobs select cute_gemm's SM120 implementations with `DLKERNEL_ARCH=120`
but compile and execute on H100 hardware, so both wheel variants work there.

## Test strategy

Per `gpu-test/action.yml`: a single pass with async kernel compilation —

- `CUDA_VISIBLE_DEVICES=$FREE_GPUS pytest tests/ -n $NUM_GPUS --dist worksteal --async-compile=24`
  (free-memory threshold 50 GB). The action waits up to 5 minutes, polling
  every 15 seconds, when runner assignment races with transient GPU
  contention. Cold kernel-compile misses are shipped to a pool of 24 CPU
  workers (forkserver sidecar, GPU-blind) while the affected tests defer and
  retry once their `.o` lands; warm runs pay nothing. The
  persistent artifact cache (`DLKERNEL_ARTIFACT_CACHE_DIR`) carries `.o` files
  across runs on the same runner. `DLKERNEL_CACHE_DIR` holds autotune winner
  JSON separately. CI removes interrupted `.o.tmp.*` exports older than 1 day
  from the artifact directory. Fingerprint-directory GC needs a policy that
  accounts for active readers and compilers; directory mtime is not a hit age.

## SIF caching on runners

The action pulls `docker://$IMAGE` into `${CI_WORK_DIR:-$HOME}/<tagslug>.sif`
on first use, then reuses the cached file on subsequent jobs with the same
tag. After each pull, **stale SIFs from previous image bumps are auto-deleted**;
the cleanup whitelist keeps both currently-pinned variants (`IMAGE_CU129` and
`IMAGE_CU132`), so cu129 and cu132 don't thrash each other's caches. Only files
matching the `<namespace>-dl-kernels-<tag>-<date>.sif` slug (plus the legacy
pre-rename `*quack-kernels*.sif` glob) are candidates, so hand-pulled SIFs like
`~/dl-kernels.sif` survive.

## Cutting a new image

The image tags are **pinned in `.github/workflows/_test.yml`** (used by both
ci.yml and ci-pr.yml). Three steps:

```bash
# 1. Build & push from a box that has docker (one-time Hub login: `docker login -u $DOCKERHUB_USER`)
./tools/ci/docker/build.sh --image --push
# On a runner, add --sif to also pre-build the SIFs the gpu-test action caches
# (rename them to .sif.hold until step 3 lands, or the action's prune deletes
# them). On a docker-less runner (b300), pre-warm from the pushed images with:
#   DATE=YY.MM.DD ./tools/ci/docker/build.sh --sif
# unattended variant: DOCKERHUB_TOKEN=hub_xxx ./tools/ci/docker/build.sh --image --push
```

```yaml
# 2. Bump IMAGE_CU129 and IMAGE_CU132 in .github/workflows/_test.yml:
env:
  IMAGE_CU129: ${{ vars.DLKERNEL_IMAGE_REPO }}:cu12.9-NEW_DATE
  IMAGE_CU132: ${{ vars.DLKERNEL_IMAGE_REPO }}:cu13.2-NEW_DATE
```

```bash
# 3. Commit and push.
git commit -am "Bump CI images to cu*-NEW_DATE"
git push
```

That's it — runners auto-pull the new SIFs on the next CI run and prune the old ones. No manual runner steps.

## Manual / local SIF testing (off-CI)

For ad-hoc debugging on a runner, pull the same image CI uses:

```bash
# $OWNER is CI's namespace: the DLKERNEL_IMAGE_REPO repository variable
apptainer pull ~/dl-kernels.sif docker://$OWNER/dl-kernels:cu12.9-DATE
apptainer exec --nv --writable-tmpfs ~/dl-kernels.sif bash
```

For private images, set `APPTAINER_DOCKER_USERNAME=<dockerhub-user>` and
`APPTAINER_DOCKER_PASSWORD=$DOCKERHUB_TOKEN` before running.

## Public vs private image (Docker Hub)

The current `<owner>/dl-kernels` repo is intended to be public, so CI needs no
Docker Hub secrets. If you flip it to private, add `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN` as repo secrets and prepend a login step before the
gpu-test action:

```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

Then export `APPTAINER_DOCKER_USERNAME` / `APPTAINER_DOCKER_PASSWORD` for the
apptainer pull step (or set them on the runner host).
