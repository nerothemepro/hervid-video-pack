# HerVid Operations Guide

Tài liệu vận hành toàn diện cho hệ thống HerVid — gen video AI trên RTX 3090
với stack: ComfyUI + LTX-2.3 (22B) + Flux keyframe + RIFE + CodeFormer + Hermes/Telegram.

---

## Mục lục

1. [Hardware & Constraints](#1-hardware--constraints)
2. [Architecture](#2-architecture)
3. [File Structure — Quy tắc 3-Copy](#3-file-structure--quy-tắc-3-copy)
4. [Startup Procedure](#4-startup-procedure)
5. [LLM Setup tối ưu](#5-llm-setup-tối-ưu)
6. [Preview Workflow](#6-preview-workflow)
7. [Mode Presets](#7-mode-presets)
8. [Quality Pipeline: Keyframe → RIFE → CodeFormer](#8-quality-pipeline-keyframe--rife--codeformer)
9. [Lỗi thường gặp & Cách fix](#9-lỗi-thường-gặp--cách-fix)
10. [GPU/VRAM Management](#10-gpuvram-management)
11. [Maintenance Checklist](#11-maintenance-checklist)

---

## 1. Hardware & Constraints

| Thành phần | Spec |
|---|---|
| GPU | RTX 3090, 24GB VRAM |
| Model LTX-2.3 | ~22GB VRAM khi render |
| Model Flux keyframe | ~7GB VRAM khi gen |
| LLM (gemma-4-12b-qat) | ~8GB VRAM khi loaded |

**Quy tắc VRAM cứng:**
- LTX-22B + LLM-8B = 30GB → **vượt quá 24GB** → OOM hoặc stall
- Trước mỗi render, LLM **phải** được eject khỏi GPU
- Flux keyframe + LTX-22B không thể cùng lúc → pipeline gọi ComfyUI `/free` giữa hai bước
- LM Studio trên host này **KHÔNG tự evict** — nó stack models cho đến khi VRAM đầy

---

## 2. Architecture

```
User (Telegram, tiếng Việt)
    ↓
Hermes Gateway  [HERMES_HOME=/opt/data/hermes-profiles/hervid]
    ↓ tool: generate_hervid_video
tools.py handler
    ├── (1) Nếu không có preview_id → auto-gọi generate_hervid_preview
    │         POST localhost:8501/generate-preview
    │         → Flux keyframe job → ảnh → gửi user confirm
    └── (2) Sau khi user confirm → gọi generate_hervid_video(preview_id=...)
              POST localhost:8501/generate-video
orchestrate.py [:8501]
    ├── step 1: LM Studio load gemma-4-12b-qat (JIT nếu chưa loaded)
    │           POST /v1/chat/completions → {prompt, character_note, duration, animation}
    ├── step 2: Eject LLM khỏi VRAM (POST /api/v1/models/unload)
    │           POST localhost:8500/generate-sequence → pipeline_api
    ├── step 3: Poll GET localhost:8500/job/{id} mỗi 30s
    │           timeout=60s mỗi poll request; retry tối đa 5 lần transient error
    └── step 4: Return {final_video_path, ...}
pipeline_api.py [:8500]
    _build_command() FORCE: --style realistic, --continuity independent
    → generate_ltx_video_sequence.py (subprocess)
        → generate_ltx_video.py (mỗi shot)
            ├── Flux keyframe 1152×768 (realistic) / 768×512 (test)
            ├── ComfyUI /free
            ├── LTX I2V → 8fps video
            ├── RIFE x3 → 24fps
            └── CodeFormer (nếu LTX_FACE_RESTORE=1 và KHÔNG phải animation)
```

**Operator rule:**
- `generate_hervid_video` = default Telegram/user-facing path
- `generate_ltx_video` = direct/operator path only, used for bounded smoke or low-level debugging
- Nếu gọi nhầm `generate_ltx_video` cho luồng user-facing, model sẽ bỏ qua bước preview và khi fail thường chỉ trả về lỗi kỹ thuật khó đọc hơn

---

## 3. File Structure — Quy tắc 3-Copy

**Mỗi file tồn tại ở 2-3 nơi — phải cập nhật TẤT CẢ khi sửa.**

### orchestrate.py (2 copies)
| Copy | Path |
|---|---|
| Source repo (git) | `/workspace/hervid-video-pack/core/orchestrate.py` |
| Live copy | `/workspace/hermes-agent/orchestrate.py` |

Sync: `cp /workspace/hervid-video-pack/core/orchestrate.py /workspace/hermes-agent/orchestrate.py`

### tools.py và __init__.py (2 copies)
| Copy | Path |
|---|---|
| Source repo | `/workspace/hermes-agent-plugin/hermes-plugin/local_media/tools.py` |
| Live copy | `/workspace/hermes-agent/plugins/local_media/tools.py` |

### Media pipeline scripts (2 copies)
| Script | Source repo | Live |
|---|---|---|
| Multi-shot sequence | `/workspace/hermes-agent-plugin/media-pipeline/generate_ltx_video_sequence.py` | `/workspace/projects/media-pipeline/generate_ltx_video_sequence.py` |
| Single-shot render | `/workspace/hermes-agent-plugin/media-pipeline/generate_video.py` | `/workspace/projects/media-pipeline/generate_video.py` |

**Lưu ý quan trọng:**
- `orchestrate.py` và `pipeline_api.py` cần **restart service** sau khi sửa
- Scripts `.py` (generate_ltx_*) dùng ngay lập tức — được gọi dưới dạng subprocess, không cần restart
- `tools.py` cần **restart Hermes gateway** sau khi sửa

### Env files (2 copies — đều cần cập nhật)
- `/opt/data/hermes/media-pipeline.env`
- `/opt/data/hermes-profiles/hervid/media-pipeline.env`

### Logs
| Service | Log path |
|---|---|
| Pipeline API (:8500) | `/tmp/hervid-logs/pipeline_api.log` hoặc `/tmp/pipeline_api.log` |
| Orchestrate (:8501) | `/tmp/orchestrate.log` |
| Hermes gateway (HerVid) | `/opt/data/hermes-profiles/hervid/logs/gateway.log` |
| Hermes gateway (default) | `/opt/data/hermes/logs/gateway.log` |

> **CRITICAL:** HerVid dùng profile `hervid`, KHÔNG phải profile default. Log ở hai nơi khác nhau. Check sai log sẽ không thấy gì cả.

---

## 4. Startup Procedure

### Khởi động bình thường (mỗi container lifecycle)
```bash
# 1. Start pipeline_api + orchestrate
bash /workspace/hervid-video-pack/scripts/start-services.sh

# 2. Start Hermes gateway với đúng HERMES_HOME
python3 /workspace/hervid-video-pack/scripts/launch-hermes.py
```

`launch-hermes.py` set `HERMES_HOME=/opt/data/hermes-profiles/hervid` tự động. Nếu khởi động Hermes không qua script này, gateway sẽ đọc `~/.hermes/` và báo "No messaging platforms enabled".

### Restart tất cả
```bash
bash /workspace/hervid-video-pack/scripts/start-services.sh --restart
python3 /workspace/hervid-video-pack/scripts/launch-hermes.py
```

### Restart chỉ orchestrate (sau khi sửa code)
```bash
kill $(lsof -ti:8501) 2>/dev/null || true
HERMES_HOME=/opt/data/hermes /workspace/.venvs/hermes-agent/bin/python \
  /workspace/hervid-video-pack/core/orchestrate.py serve \
  > /tmp/orchestrate.log 2>&1 &
```

### Verify services healthy
```bash
curl -s http://localhost:8500/health | python3 -m json.tool
curl -s http://localhost:8501/health
```

### Python binary bắt buộc
Luôn dùng `/workspace/.venvs/hermes-agent/bin/python` — KHÔNG dùng `python3` system.

---

## 5. LLM Setup tối ưu

### Model được chọn: `google/gemma-4-12b-qat`

**Kết quả A/B test (2026-06-20):**
| Model | Thời gian | Chất lượng | Lý do chọn/loại |
|---|---|---|---|
| gemma-4-12b-qat | 58s | ★★★★★ | **WINNER** — 25 tok/s, continuity tốt nhất, no thinking-tax |
| qwen3.6-27b | 219s (3.7 phút!) | ★★★★ | LOẠI — thinking-tax không tắt được trong build này |

**Vấn đề với qwen3.6:** Không thể tắt thinking mode dù đã dùng `/no_think` hoặc `chat_template_kwargs:{enable_thinking:false}`. Luôn burn ~2400 reasoning tokens trước khi trả lời. Ở 14 tok/s = 3-4 phút/shot-list. Không dùng được cho Telegram realtime.

### Config LM Studio cho orchestrate
```yaml
# /opt/data/hermes-profiles/hervid/config.yaml
model:
  default: google/gemma-4-12b-qat
  provider: lmstudio
  context_length: 65536
```

### Gemma quirk — max_tokens=800 (bắt buộc)
Gemma có reasoning phase (`reasoning_content`). Với `max_tokens=300`, toàn bộ token budget bị dùng cho thinking → `content` trả về rỗng. Phải dùng `max_tokens=800`.

### LM Studio URL — cần `/v1` suffix
```python
# orchestrate.py — normalize URL
_LM_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
LM_URL = _LM_BASE.rstrip("/")
if not LM_URL.endswith("/v1"):
    LM_URL = LM_URL + "/v1"
LM_MGMT = LM_URL[: LM_URL.rfind("/v1")]  # http://host.docker.internal:1234
```

Nếu thiếu `/v1`: LM Studio trả lỗi `Unexpected endpoint or method (POST /chat/completions)` → orchestrate retry 3 lần với backoff [15, 30]s → **tổng ~45s → 500 error**.

### LM Studio model unload (working endpoint)
```bash
# GET list instances
curl http://host.docker.internal:1234/api/v1/models

# POST unload (endpoint DUY NHẤT hoạt động)
curl -X POST http://host.docker.internal:1234/api/v1/models/unload \
  -H 'Content-Type: application/json' \
  -d '{"instance_id": "<id_từ_response_trên>"}'
```

**KHÔNG dùng:** `/api/v0/models/unload`, `/v1/models/unload`, DELETE — tất cả return 200 nhưng không unload gì cả.

---

## 6. Preview Workflow

HerVid dùng 2-bước workflow để tránh lãng phí 8-60 phút render video sai composition:

```
[1] generate_hervid_video(brief="...")
        ↓ handler auto-gọi preview (không cần preview_id)
    POST /generate-preview → Flux keyframe ~40-180s (có thể lâu hơn khi ComfyUI đang bận)
        ↓ gửi ảnh lên Telegram
    "Bố cục OK chưa?"

[2] User: "OK, gen video đi"
    generate_hervid_video(preview_id="abc123", mode="quality")
        ↓ orchestrate lấy creative từ _preview_store[preview_id]
    Video render full
```

**Tại sao enforce trong handler:** Gemma-4-12b không đủ mạnh để follow instruction "call preview first". Hard-enforce trong `handle_generate_hervid_video` — nếu không có `preview_id`, tool tự chạy preview và return ảnh. Model không cần "hiểu" workflow.

**Timeout hiện tại:**
- preview timeout mặc định ở orchestrate đã được nới thành `900s`
- lý do: log thực tế cho thấy có job Flux keyframe hoàn tất sau khi ngưỡng `300s` cũ đã bị vượt qua
- nếu ComfyUI đang có queue hoặc GPU vừa recover sau render nặng, preview có thể chậm hơn mức 1-2 phút lý tưởng

**Preview store:** In-memory dict trong orchestrate (`_preview_store`). **Bị xóa khi restart orchestrate.** Sau restart, user phải gen keyframe lại từ đầu.

**Mode behavior khi dùng preview_id:**
- Mode trong request luôn được tôn trọng (wins over stored mode)
- Nếu không pass mode → default "quality"

**Config:**
```yaml
# /opt/data/hermes-profiles/hervid/config.yaml
max_turns: 15  # phải >= 15 để đủ turns cho full flow
```

---

## 7. Mode Presets

Tất cả mode đều dùng **8fps native** (RIFE sau đó x3 → 24fps cho standard/quality).

| Mode | Resolution | Steps | Shot duration | Frames/shot | Approx time/shot | Dùng khi |
|---|---|---|---|---|---|---|
| `test` | 512×320 | 1 | 1s | 9 | ~76s | Debug, kiểm tra pipeline |
| `standard` | 512×320 | 12 | 3s | 25 | ~376s | Draft review |
| `quality` | 768×512 | 26 | 3s | 25 | ~400-600s | Production |

**Keyframe resolution theo mode:**
| Mode | Keyframe | Flux steps |
|---|---|---|
| test | 768×512 | 8 |
| standard / quality | 1152×768 | 28 |

**30s video ở quality mode:** ~10 shots × 400-600s = **~1-1.5 giờ**.

---

## 8. Quality Pipeline: Keyframe → RIFE → CodeFormer

### 8.1 Keyframe Upscale (Flux)

**Vấn đề:** Keyframe gen ở 832×480 → face chỉ chiếm vài pixel → LTX không thể thêm chi tiết.

**Fix:** Realistic keyframe ở 1152×768 (28 steps) + face-positive tokens:
```
"detailed facial features, sharp eyes, natural skin texture, fine detail"
```

**NEGATIVE_PROMPT** phải cập nhật trong env file (KHÔNG chỉ trong code):
```bash
# /opt/data/hermes-profiles/hervid/media-pipeline.env
NEGATIVE_PROMPT=blurry, low quality, distorted, deformed, ugly, ..., blurry face, unrealistic face
```

### 8.2 RIFE Frame Interpolation (24fps)

**Vấn đề:** LTX render 8fps → choppy, blurry motion.

**Fix:** RIFE x3 trong ComfyUI graph (sau VAEDecodeTiled, trước CreateVideo):
- `rife_v4.26.safetensors` đã có trong container
- Node: `FrameInterpolationModelLoader` + `FrameInterpolate` (multiplier=3)
- test mode: interp OFF (multiplier=1)
- standard/quality: multiplier=3 → 24fps output

RIFE không tốn VRAM thêm (chạy sau decode).

### 8.3 CodeFormer Face Restore

**Vấn đề:** Face vẫn blur DURING action vì RIFE không un-blur frame đã smear.

**Fix:** Per-frame CodeFormer trước RIFE (ít frame hơn = nhanh hơn):
```
VAEDecodeTiled → FaceRestoreCFWithModel → FrameInterpolate → CreateVideo
```

**Env gate:**
```bash
LTX_FACE_RESTORE=1      # 1=on, 0=off (default=0 nếu node chưa install)
LTX_FACE_FIDELITY=0.5   # 0=max restore, 1=max input fidelity
```

**Auto-disable cho animation:** CodeFormer tự OFF khi prompt chứa `pixar/cartoon/anime/hoạt hình`. Retinaface không detect stylized face.

**Install CodeFormer** (chạy trên HOST, không phải sandbox):
```bash
docker exec gen-media-comfy bash -lc '
  set -e
  ROOT=$(dirname "$(find / -maxdepth 6 -type d -name custom_nodes 2>/dev/null | head -1)")
  cd "$ROOT/custom_nodes"
  [ -d facerestore_cf ] || git clone https://github.com/mav-rik/facerestore_cf.git
  pip install -r facerestore_cf/requirements.txt || pip install facexlib
  mkdir -p "$ROOT/models/facerestore_models"
  cd "$ROOT/models/facerestore_models"
  [ -f codeformer.pth ] || wget -O codeformer.pth \
    https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
  echo DONE
'
docker restart gen-media-comfy
```

### 8.4 Shot Duration vs Drift

**Vấn đề (anchor drift):** Shot 5s = 41 frames nhưng chỉ frame 0 conditioned trên keyframe → face drift/morph sau giữa shot.

**Fix:** Shot 3s (25 frames) → drift window -40%.

Đã đo bằng `ffmpeg blurdetect`: blur tăng đều từ pos0 → end trong mỗi shot, reset sharp tại shot boundary → xác nhận là drift, không phải RIFE.

---

## 9. Lỗi thường gặp & Cách fix

### 9.1 step3_poll timeout → false "failed" (video đã render xong)

**Triệu chứng:** Orchestrate báo lỗi `Read timed out. (read timeout=15)` sau 15s poll. Video thực ra đã render xong (check `/opt/data/hermes/generated-videos/`).

**Root cause:** GPU render nặng → uvicorn pipeline_api delay >15s responding `/job/{id}` → timeout.

**Fix hiện tại:** timeout=60s + retry tối đa 5 lần consecutive errors (trong `step3_poll`).

**Nếu vẫn xảy ra:** Tăng `MAX_WAIT` trong orchestrate.py hoặc check pipeline_api còn sống không:
```bash
curl -s http://localhost:8500/health
```

### 9.2 LLM mode override (user nói "test" nhưng render "quality")

**Triệu chứng:** Render mất 37 phút dù user yêu cầu mode=test.

**Root cause cũ:** `mode = stored.get("mode", mode)` → stored preview mode ghi đè request mode.

**Fix:** `if not mode: mode = stored.get("mode", "quality")` → request mode luôn wins.

### 9.3 OOM tại KSampler (fp8 stochastic rounding)

**Triệu chứng:** Render fail ở bước KSampler dù còn đủ VRAM (24GB free). Không phải do model size.

**Root cause:** ComfyUI fp8 stochastic rounding allocate 5-7 temp tensors lớn per weight → peak VRAM vượt 24GB.

**Fix** (apply sau mỗi container recreate):
```bash
docker exec gen-media-comfy sed -i \
  's/stochastic_rounding=seed/stochastic_rounding=0/' \
  /opt/ComfyUI/comfy/ops.py
docker restart gen-media-comfy
```

> **CRITICAL:** Patch này nằm trong container writable layer, **mất khi recreate container**. Phải re-apply.

### 9.4 GPU bad memory state (OOM dù VRAM "free")

**Triệu chứng:** Ngay cả mode=test (512×320, 1 step) cũng OOM hoặc stall >10 phút. ComfyUI `/system_stats` báo ~24GB free.

**Root cause:** Sau nhiều OOM, GPU VRAM fragmented → `cudaMallocAsync` báo free nhưng không có block contiguous đủ lớn. ComfyUI restart KHÔNG đủ vì LM Studio còn CUDA context.

**Fix:**
1. Eject tất cả LM Studio models (UI hoặc `lms unload --all` trên host)
2. `docker restart gen-media-comfy`
3. Nếu vẫn lỗi: reboot host
4. Confirm bằng mode=test render → phải xong trong ~76s

### 9.5 Orchestrate 45s timeout (LM Studio URL hoặc stuck threads)

**Triệu chứng:** Mọi request đều fail sau đúng ~45s.

**Nguyên nhân 1 — URL thiếu `/v1`:**
Orchestrate retry 3 lần × [0+15+30]s = 45s tổng. Check:
```bash
grep LM_STUDIO_BASE_URL /opt/data/hermes-profiles/hervid/.env
```
Sửa: đảm bảo URL kết thúc bằng `/v1` (orchestrate.py tự normalize nhưng env var phải đúng).

**Nguyên nhân 2 — Stuck threads:**
`curl --max-time 5` → client disconnect → server thread vẫn hold LM Studio connection. Tất cả request mới instant-fail.
Fix: restart orchestrate.

### 9.6 Shot boundary failure (multi-shot sequence)

**Triệu chứng:** Shot 1 xong nhưng shot 2 fail với `empty_queue_seconds=120`.

**Root cause:** ComfyUI queue drop transient sau khi xử lý xong shot trước.

**Fix đã có:** `generate_ltx_video_sequence.py` tự:
1. POST ComfyUI `/free` + 6s cooldown trước mỗi shot sau shot 1
2. Retry 1 lần nếu fail (sau free+cooldown)

Nếu vẫn fail sau retry → check VRAM, check ComfyUI logs.

### 9.7 "No messaging platforms enabled"

**Nguyên nhân:** Hermes gateway khởi động với `~/.hermes/` thay vì `/opt/data/hermes-profiles/hervid/` → không tìm thấy Telegram config.

**Fix:** Luôn dùng `launch-hermes.py` script để start gateway:
```bash
python3 /workspace/hervid-video-pack/scripts/launch-hermes.py
```

### 9.8 Wrong tool routing (Wan2.1 thay vì LTX)

**Triệu chứng:** Prompt animation/cartoon → render ra "samurai bamboo forest" style.

**Root cause:** Wan2.1 pipeline `generate_video_sequence.py` có hardcoded storyboard "two samurai in bamboo forest" khi `style=anime_action`. Gemma route prompt animation về tool sai.

**Fix:** tools.py description hiện đánh dấu LTX là default cho TẤT CẢ styles, Wan là legacy.

**Sau restart chưa có:** Nói user dùng "tool generate_ltx_video_sequence" hoặc "LTX" explicit.

### 9.9 tools.py thay đổi không có effect

**Nguyên nhân:** Tools.py load lúc gateway start, không hot-reload.

**Fix:** Restart Hermes gateway sau bất kỳ thay đổi nào ở tools.py hoặc __init__.py.

---

## 10. GPU/VRAM Management

### Trước mỗi render

Pipeline tự động:
1. `_lm_eject()` trong orchestrate → POST `/api/v1/models/unload` cho từng model
2. ComfyUI `/free` trong generate_ltx_video.py trước LTX load
3. 2s sleep sau `/free`

Gated bởi env: `LTX_AUTO_EJECT_LLM=1` (default ON).

### Check VRAM thủ công
```bash
# Từ host
nvidia-smi

# ComfyUI stats (từ sandbox/container)
curl http://host.docker.internal:8188/system_stats | python3 -m json.tool
```

### Sau render — LLM auto-reload

Khi user chat tiếp sau render, LM Studio JIT-load gemma-4-12b lại (~54s nếu từ disk cold). Đây là behavior bình thường.

### VRAM budget summary

| State | VRAM used | OK? |
|---|---|---|
| Idle (LM Studio off) | ~0GB | ✅ |
| gemma-4-12b loaded | ~8GB | ✅ |
| Flux keyframe render | ~7GB | ✅ |
| LTX-22B render (quality) | ~22GB | ✅ (barely) |
| gemma + LTX cùng lúc | ~30GB | ❌ OOM |
| Multiple LLMs stacked | 16-24GB+ | ❌ |

---

## 11. Maintenance Checklist

### Sau container recreate (`gen-media-comfy`)
- [ ] Re-apply fp8 stochastic rounding patch:
  ```bash
  docker exec gen-media-comfy sed -i 's/stochastic_rounding=seed/stochastic_rounding=0/' /opt/ComfyUI/comfy/ops.py
  docker restart gen-media-comfy
  ```
- [ ] Verify CodeFormer node vẫn còn: `curl http://host.docker.internal:8188/object_info | python3 -m json.tool | grep FaceRestore`
- [ ] Nếu mất → reinstall (xem mục 8.3)
- [ ] Test render với mode=test → phải xong ~76s

### Sau code change
- [ ] Nếu sửa orchestrate.py → sync cả 2 copy → restart service (:8501)
- [ ] Nếu sửa tools.py → sync cả 2 copy → restart Hermes gateway
- [ ] Nếu sửa generate_ltx_*.py → sync cả 2 copy → không cần restart (subprocess)
- [ ] Nếu sửa env files → cả 2 env files → restart pipeline_api nếu cần

### Health check nhanh
```bash
# Pipeline API
curl -s http://localhost:8500/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if d['ok'] else 'FAIL', d)"

# Orchestrate
curl -s http://localhost:8501/health

# ComfyUI
curl -s http://host.docker.internal:8188/system_stats | python3 -c "import sys,json; d=json.load(sys.stdin); print('ComfyUI OK, VRAM free:', d.get('system',{}).get('vram_free',0)//1024//1024, 'MB')"

# LM Studio
curl -s http://host.docker.internal:1234/api/v1/models | python3 -m json.tool
```

### Test pipeline end-to-end (nhanh nhất)
```bash
curl -X POST http://localhost:8501/generate-preview \
  -H 'Content-Type: application/json' \
  -d '{"brief": "a cat sitting on a table", "mode": "test"}' \
  -w "\nHTTP %{http_code}\n"
```
Phải return sau ~40-90s với `preview_id` và `image_path`.

---

## GitHub Repos

- `nerothemepro/hervid-video-pack` — pipeline_api, orchestrate, start scripts
- `nerothemepro/hermes-agent-plugin` — Hermes tools plugin

## Commit convention
```
<summary line>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```
