# HerVid Video Pack — Plan & Architecture

> Sản phẩm: bộ workflow pack tự host để gen video ngắn (Telegram → kịch bản → ComfyUI+LTX-2.3 → hậu kỳ → video hoàn chỉnh).
> Mục tiêu: diệt root-cause "agent làm sai bước" bằng orchestration deterministic, và đóng gói bán được hợp lệ về license.
> Ngày tổng hợp: 2026-06-21.

---

## 1. Vấn đề gốc cần giải

Pipeline Hermes hiện tại để LLM (gemma-12b) quyết định **cả phần sáng tạo lẫn phần cơ học** (chọn tool, style, tham số). LLM thường xuyên sai phần cơ học:
- Routing sai (animation → Wan samurai pipeline)
- Chọn `style=anime` cho nội dung Pixar → animagine sinh nhân vật anime/samurai
- Chọn sai tham số

**Giải pháp:** tách 2 vai. LLM **chỉ** làm sáng tạo (kịch bản → shot list). Phần cơ học (tool nào, param gì) **hardcode** ở tầng orchestration → không thể sai.

---

## 2. Kết quả research (tóm tắt)

Không repo open-source nào làm sẵn "multi-shot LTX sequence orchestration trên 1 GPU 24GB". Mọi ComfyUI API wrapper đều **single-workflow-per-request**. Phần multi-shot (loop N cảnh + keyframe/cảnh + anti-drift + retry + stitch) **chỉ có trong `generate_ltx_video_sequence.py` của mình** → đây là tài sản lõi, giữ nguyên.

Các mảnh ghép tham khảo:
- **SaladTechnologies/comfyui-api** (⭐433, MIT, LTX verified) — API server bọc ComfyUI, single-shot. Tham khảo pattern, có thể dùng để scale sau.
- **ai-dock/comfyui-api-wrapper** — FastAPI 3-tầng (pre/gen/post), tham khảo cấu trúc job-queue.
- **kustomzone/ComfyUI-vidflows** — multi-shot story 4-LLM, nhưng cần 96GB VRAM + Wan14B, dùng last-frame continuation (drift nặng). Không dùng được; xác nhận thiết kế independent-keyframe của mình tốt hơn cho video dài.
- **burnsbert/ComfyUI-EBU-LMStudio** (⭐47, MIT) — gọi LM Studio trong ComfyUI; dự phòng cho v-sau.

---

## 3. Kiến trúc chốt (PA4 — Hybrid)

```
Telegram → n8n Webhook
  → [HTTP → LM Studio]   LLM CHỈ sinh shot-list JSON (không quyết tool/param)
  → [Function: validate JSON]   deterministic
  → [HTTP POST → Pipeline API /generate-sequence]   param cơ học HARDCODE
  → [Wait + poll /job/{id}]
  → [hậu kỳ: phụ đề / audio]   (v2+)
  → [HTTP → Telegram sendVideo]
```

- **Lõi (Python, tài sản của mình):** `generate_ltx_video_sequence.py` + `generate_ltx_video.py` + `generate_video.py` — đã chứa mọi fix: anti-drift (shot 3s + steps 26), block animagine cho animation, RIFE 24fps, face-restore gating, auto-eject LLM, retry shot-boundary.
- **Tầng API (mới, v1):** `pipeline_api.py` — FastAPI bọc sequence script. Param cơ học (style=realistic, engine=flux, continuity=independent) bị **ép cứng** → caller/LLM không thể chọn sai.
- **Tầng ngoài (n8n, v1.5):** Telegram I/O + node LM Studio + gọi API.

**Vì sao Hybrid tối ưu:** n8n lo phần nó giỏi (routing deterministic, diệt lỗi agent); phần multi-shot phức tạp giữ trong Python đã test kỹ → không tái sinh bug.

---

## 4. Đóng gói thương mại (Self-host pack)

### Ranh giới license (vàng)
Gói bán = **code của mình + file JSON cấu hình**. n8n & ComfyUI **luôn tải từ upstream lúc cài**, KHÔNG redistribute trong gói → né Sustainable Use License (n8n) + nghĩa vụ GPL (ComfyUI).

| Thành phần | License | Trong gói bán? |
|---|---|---|
| Code core của mình | License của mình | ✅ chứa |
| n8n workflow JSON (template) | — | ✅ chứa (bán template hợp lệ) |
| ComfyUI workflow JSON | — | ✅ chứa (config, không phải binary) |
| n8n runtime | Sustainable Use | ❌ install.sh `docker pull` từ upstream |
| ComfyUI runtime | GPL-3.0 | ❌ install.sh `git clone` từ upstream |
| LTX-2 weights | Community License (free <$10M ARR) | ❌ download_models.sh tải từ HF |
| flux1-schnell | Apache-2.0 | ❌ tải từ HF |

### Cấu trúc package
```
hervid-video-pack/
├── PLAN.md                    # tài liệu này
├── install.sh                 # cài, KHÔNG bundle n8n/ComfyUI
├── config.example.yaml        # khách điền token/URL/paths
├── LICENSE-EULA.txt           # license của mình cho core
├── THIRD-PARTY-NOTICES.md     # license bên thứ ba
├── core/                      # ★ tài sản lõi
│   ├── pipeline_api.py        # FastAPI wrap (v1)
│   ├── generate_ltx_video*.py # lõi sequence (copy từ pipeline đã test)
│   └── postprocess/           # tts, subtitle, ffmpeg (v2+)
├── comfyui-workflows/         # JSON workflow LTX/flux
├── n8n-workflows/             # JSON template import vào n8n của khách
└── scripts/
    ├── download_models.sh
    └── apply_patches.sh       # fp8 stochastic rounding, facerestore
```

### Decouple cần làm trước khi bán
- Path cứng `/opt/data`, `/workspace` → `config.yaml`
- 2-3 bản copy script → 1 package version-hoá
- env profile → `config.example.yaml`
- Patch thủ công → `apply_patches.sh`

### Giảm support burden (điểm yếu của self-host)
- `install.sh` fail sớm + báo rõ (thiếu VRAM/driver/Docker)
- `doctor.sh` chẩn đoán (GPU free? ComfyUI sống? model đủ?)
- Doc + video demo + cộng đồng Discord/Telegram thay support 1-1

---

## 5. Lộ trình phiên bản

| Ver | Phạm vi | Trạng thái |
|---|---|---|
| **v1** | `pipeline_api.py` FastAPI wrap + test curl. Param cơ học ép cứng. | ✅ XONG (verified e2e 2026-06-22: job test-mode chạy queued→running→completed, xuất final.mp4 + manifest, continuity=independent đúng) |
| v1.5 | n8n workflow JSON (Telegram → LM Studio shot-list → API → sendVideo) | kế tiếp |
| v2 | Phụ đề (Whisper → SRT → ffmpeg burn-in) | |
| v3 | Audio/TTS + nhạc nền | |
| v4 | Trending API → đề xuất nội dung | |
| v5 | Đóng gói pack đầy đủ (install.sh, decouple, doctor, EULA) | |

**Nguyên tắc:** mỗi bước thêm = một điểm fail → build & test tăng dần, không làm hết một lúc.

---

## 6. v1 — Đặc tả Pipeline API

`core/pipeline_api.py` — FastAPI, single-GPU job queue (1 render/lần).

**Giá trị cốt lõi:** param cơ học bị ép cứng ở tầng API → caller chỉ truyền nội dung sáng tạo, KHÔNG thể chọn sai style/engine (diệt bug samurai).

Endpoints:
- `POST /generate-sequence` → tạo job, enqueue, trả `job_id`
- `GET /job/{id}` → trạng thái (queued/running/completed/failed) + `final_video_path` khi xong
- `GET /jobs` → danh sách job
- `GET /health` → ping ComfyUI
- `POST /generate-sequence?validate_only=true` → smoke test nhanh (không render)

Param ÉP CỨNG (caller không override được): `style=realistic`, `keyframe_engine=auto(→flux)`, `continuity=independent`.
Param cho phép: `prompt` (bắt buộc), `mode` (quality), `total_duration_seconds`, `shot_duration_seconds` (3), `animation` (auto), `character_note`, `seed`.

Cấu hình qua env (default khớp cài đặt hiện tại):
- `HVP_SCRIPT_PATH`, `HVP_ENV_FILE`, `HVP_OUTPUT_DIR`, `HVP_COMFY_URL`, `HVP_PYTHON_BIN`
