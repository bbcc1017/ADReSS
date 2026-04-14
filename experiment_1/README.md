# experiment_1 — 한국 좌표 배치 실험

Streamlit 대시보드와 **독립적으로** 동작하는 논문 실험용 배치 파이프라인입니다.
한국 육지 경계 내 랜덤 좌표 1000개에 대해 시나리오 생성 → 시뮬레이션을 자동으로 배치 처리합니다.

---

## 폴더 구조

```
experiment_1/
├── generate_coords.py          # 좌표 생성기 (shapefile PIP 필터링)
├── batch_runner.py             # 배치 실험 러너 (시나리오 생성 + 시뮬레이션)
├── visualize_coords.py         # 배치 결과 지도·히스토그램·규칙분석 시각화
└── ctprvn.shp / .shx / .dbf   # 한국 행정구역 shapefile (좌표 생성 시 사용)
```

시각화 산출물(`coords_map.html`, `coords_map_hist.*`, `coords_map_rule_heatmap.*`, `coords_map_rule_effects.*`)과 실험 데이터(`coords.csv`, `progress.json`)는 `scenarios/{experiment_id}/` 폴더에 저장됩니다.

---

## 전제 조건

```bash
pip install geopandas shapely
```

나머지 의존성(`folium`, `pandas`, `requests` 등)은 프로젝트 루트 `requirements.txt`에 포함되어 있습니다.

---

## Step 1 — 좌표 생성

모든 명령은 **프로젝트 루트** (`MCI_ADV/`)에서 실행합니다.

```bash
python experiment_1/generate_coords.py --n 1000 --seed 0
```

완료 후:
- `experiment_1/coords_korea.csv` 생성 (컬럼: `coord_id`, `latitude`, `longitude`, `generated_at`)
- `experiment_1/coords_map.html` 생성 → 브라우저로 열어 모든 점이 육지에 있는지 확인

**옵션:**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--n` | `1000` | 생성할 좌표 수 |
| `--seed` | `0` | 랜덤 시드 |
| `--shp` | `ctprvn.shp` | shapefile 경로 (experiment_1/ 기준) |
| `--out` | `experiment_1/coords_korea.csv` | 출력 CSV 경로 |
| `--map` | `<out_dir>/coords_map.html` | folium 지도 출력 경로 |

소규모 테스트:
```bash
python experiment_1/generate_coords.py --n 50 --seed 0
```

---

## Step 2 — 배치 실험 실행

**매일 동일한 명령**을 실행하면 `progress.json`을 보고 자동으로 이어서 처리합니다.

### 기본 실행 (Kakao 모드)

```bash
python experiment_1/batch_runner.py \
    --kakao-api-key YOUR_KEY \
    --experiment-id exp_batch_research
```

### OSRM 모드 (카카오 키 불필요, 오픈소스 백엔드)

카카오 API 키가 없는 경우 `--is-use-time false`를 주면 [OSRM](https://project-osrm.org/docs/v5.24.0/api/#) HTTP API로 도로 거리/시간을 받아 카카오와 동일한 스키마로 저장된다. 시뮬레이터는 `distance/velocity`로 시간을 산출하며, 동일 시나리오 폴더로 시뮬을 재실행할 때 YAML의 `is_use_time`을 `True`로 바꾸면 OSRM duration 기반 시뮬도 가능하다.

```bash
# 데모 서버 사용 (소규모 테스트 한정)
python experiment_1/batch_runner.py \
    --is-use-time false \
    --experiment-id exp_batch_osrm

# 자체 호스팅 OSRM (운영 권장)
python experiment_1/batch_runner.py \
    --is-use-time false \
    --osrm-url http://localhost:5000 \
    --experiment-id exp_batch_osrm
```

자체 호스팅 도커 절차는 프로젝트 루트 `README.md`의 "OSRM 백엔드" 섹션 참고.

### 파라미터 전체 지정 예시

```bash
python experiment_1/batch_runner.py \
    --kakao-api-key YOUR_KEY \
    --experiment-id exp_batch_research \
    --daily-limit 5000 \
    --calls-per-coord 40 \
    --incident-size 30 \
    --amb-count 30 \
    --uav-count 3 \
    --amb-velocity 40 \
    --uav-velocity 80 \
    --total-samples 30 \
    --random-seed 0 \
    --amb-handover-time 10.0 \
    --uav-handover-time 15.0 \
    --is-use-time true \
    --duration-coeff 1.0 \
    --hospital-max-send-coeff "1.0,1.0" \
    --departure-time 202604071400
```

### 파라미터 설명

**경로 / 실험 식별**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--base-path` | 프로젝트 루트 | MCI_ADV 루트 경로 |
| `--coords` | `experiment_1/coords_korea.csv` | 좌표 CSV |
| `--progress` | `experiment_1/progress.json` | 진행 상태 파일 |
| `--experiment-id` | `exp_batch_research` | 실험 ID (`scenarios/` 하위 폴더명) |

**시나리오 / 시뮬레이션 파라미터**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--incident-size` | `30` | 환자 수 |
| `--amb-count` | `30` | 구급차 수 |
| `--uav-count` | `3` | UAV 수 |
| `--amb-velocity` | `40` | 구급차 속도 (km/h) |
| `--uav-velocity` | `80` | UAV 속도 (km/h) |
| `--total-samples` | `30` | 시뮬레이션 반복 횟수 |
| `--random-seed` | `0` | 랜덤 시드 |

**이송 / 시간 파라미터**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--amb-handover-time` | `0.0` | 구급차 환자 인계시간 (분) |
| `--uav-handover-time` | `0.0` | UAV 환자 인계시간 (분) |
| `--is-use-time` | `true` | `true`: 카카오 API duration / `false`: OSRM 정적 거리(distance/velocity). `false` 모드에서도 OSRM duration이 CSV에 저장되어 추후 `is_use_time=true`로 재실행 시 활용 가능 |
| `--duration-coeff` | `1.0` | API duration 시간 가중치 |
| `--osrm-url` | env `MCI_OSRM_URL` 또는 `https://router.project-osrm.org` | OSRM HTTP API base URL (`is-use-time=false` 전용). 데모 서버는 fair-use 정책 있음 — 자체 호스팅 권장 |

**병원 할당 파라미터**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--hospital-max-send-coeff` | 내부 기본값 | 병원 전송계수 `"1.0,1.0"` 형식 |
| `--buffer-ratio` | 내부 기본값 | 후보 병원 버퍼 배수 |
| `--util-by-tier` | 내부 기본값 | 등급별 이용률 `"1:0.90,11:0.75,etc:0.60"` 형식 |

**API / 배치 제어**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--kakao-api-key` | — | Kakao REST API 키 (`--is-use-time true` 모드에서만 필수, OSRM 모드에서는 불필요) |
| `--departure-time` | 현재 시간 | Kakao 출발시간 `YYYYMMDDHHmm` 형식 (선택) |
| `--daily-limit` | `5000` | 하루 최대 API 호출 수 (Kakao 일일 한도: 5000) |
| `--calls-per-coord` | `40` | 좌표 1개당 예상 API 호출 수 |
| `--max-retries` | `2` | 실패 좌표 최대 재시도 횟수 |

---

## 진행 현황 확인

```bash
python experiment_1/batch_runner.py --status
```

출력 예시:
```
=== 진행 현황 (2026-03-18 14:30) ===
완료    : 188 / 1000  (18.8%)
  ├ 시뮬 성공 : 185
  └ 시뮬 실패 : 3
오류    : 5
미처리  : 807
오늘 사용 API: 4895 / 4900
오늘 잔여 예산: 5콜 → 약 0개 처리 가능
최근 처리: coord_id=188 @ 2026-03-18 13:55
```

`--status`는 `--progress` 인수만 있으면 동작합니다. Kakao API 키 불필요.

---

## 기타 실행 모드

**dry-run** — API 호출 없이 오늘 처리 예정 목록과 예산 계산만 출력:
```bash
python experiment_1/batch_runner.py \
    --kakao-api-key YOUR_KEY \
    --dry-run
```

**소량 테스트** — 실제 2개 좌표만 실행:
```bash
python experiment_1/batch_runner.py \
    --kakao-api-key YOUR_KEY \
    --daily-limit 110 \
    --calls-per-coord 55
```

---

## 자동화 흐름

```
매일 동일 명령 실행
       │
       ▼
progress.json 로드 (없으면 신규 생성)
       │
       ▼
오늘 잔여 API 예산 계산 → 처리 가능 개수 결정
       │
       ▼
pending / failed(재시도 가능) 좌표 순서대로 처리
  │
  ├─ generate_scenario()  ← Kakao API 호출
  │    ├─ 성공 → 다음 단계
  │    └─ 실패 (API 오류, 타임아웃 등)
  │         → failed 처리, 다음 세션에서 자동 재시도
  │
  └─ run_simulation()     ← API 호출 없음
       ├─ 성공 → done (sim_ok=true)
       └─ 실패 → done (sim_ok=false), 원인 기록
       │
       ▼
  progress.json 즉시 저장 (크래시 안전)
       │
       ▼
예산 소진 또는 전부 완료 → 종료
```

### API 제한 처리 방식

- 실행 전에 `(잔여 예산) ÷ (좌표당 호출 수)`로 처리 가능 개수를 계산해 초과 시작을 방지
- 시나리오 생성 중 API 오류(429, 타임아웃 등)로 subprocess가 비정상 종료되면 해당 좌표를 `failed` 처리
- `failed` 좌표는 `attempts < max-retries` 조건 내에서 다음 세션에 자동 재시도
- `max-retries` 초과 시 `abandoned`로 전환 (더 이상 시도하지 않음)
- Ctrl+C로 중단해도 `progress.json`에 현재까지 결과가 저장됨

---

## progress.json 구조

```json
{
  "experiment_id": "exp_batch_research",
  "total": 1000,
  "statuses": {
    "1":  {"status": "done",    "config_path": "...", "sim_ok": true,  "finished_at": "..."},
    "2":  {"status": "failed",  "step": "generate",  "attempts": 1,   "error": "..."},
    "3":  {"status": "pending"}
  },
  "api_log": [
    {"date": "2026-03-18", "calls_used": 4895, "coords_processed": 89}
  ]
}
```

**status 값:**

| 값 | 의미 |
|----|------|
| `pending` | 미처리 |
| `running` | 처리 중 (비정상 종료 시 이 상태로 남을 수 있음 → 재실행 시 재시도) |
| `done` | 시나리오 생성 완료 (`sim_ok` 필드로 시뮬레이션 성공 여부 구분) |
| `failed` | 오류 발생, `attempts < max-retries`이면 다음 세션에서 재시도 |

---

## 결과 위치

시나리오 생성 결과:
```
scenarios/exp_batch_research/(lat,lon)/
├── config_(lat,lon).yaml
├── patient_info.csv
├── hospital_info_road.csv
├── amb_info_road.csv
├── uav_info.csv
└── ...
```

시뮬레이션 결과:
```
results/exp_batch_research/(lat,lon)/
├── results_(lat,lon).txt       # RAW 데이터
└── results_(lat,lon)_stat.txt  # 통계 요약
```

실행 로그:
```
experiment_logs/(lat,lon)_YYYYMMDD_HHMMSS.txt
```

---

## Step 3 — 결과 시각화

```bash
python experiment_1/visualize_coords.py
```

완료 후 `scenarios/{experiment_id}/` 폴더에 시각화 파일들이 생성됩니다. 대시보드 BatchExperiment 페이지에서도 동일한 결과를 확인할 수 있습니다.

### 산출물

| 파일 | 설명 |
|------|------|
| `coords_map.html` | 인터랙티브 결과 지도 (단일 HTML) |
| `coords_map_hist.pdf/png` | 지표별 히스토그램 |
| `coords_map_rule_heatmap.pdf/png` | 64개 규칙 성능 히트맵 |
| `coords_map_rule_effects.pdf/png` | 요인별 주효과(Main Effects) 그래프 |

### 기능

- **결과 지도 (Leaflet.js)**
  - Reward / Time / PDR 3개 지표를 상단 버튼으로 전환
  - **지도 타일 전환**: OpenStreetMap / CartoDB 두 가지 타일 선택 가능
  - RdYlGn 컬러맵: Reward는 높을수록 초록, Time·PDR은 낮을수록 초록
  - P5~P95 백분위수 클리핑: 극단값에 의한 색상 포화 방지
  - 이상치 강조: 상위·하위 N개 좌표를 별도 색상(파랑/보라)으로 표시
  - 시나리오 생성 실패 좌표: 검은색 마커 (토글로 표시/숨김)
  - 좌측 범례: 컬러바 + 이상치 목록(`<details>` 패널 접기/펼치기)
  - 줌 세밀 조절 (0.5단계 줌)
- **히스토그램 (matplotlib)**
  - Freedman-Diaconis 빈 크기 (min 60, max 120개 빈)
  - 이상치 구간 빈을 별도 색상으로 칠하고 rug plot 추가
  - 평균·표준편차 통계 박스
- **규칙 히트맵**: 3지표 × 4패널(Priority×HosSelect) 매트릭스, 각 패널은 4×4(Red Mode×Yellow Mode) 히트맵
- **주효과 그래프**: 4개 요인(Priority, HosSelect, Red Mode, Yellow Mode)별 marginal mean 막대그래프, Best level 빨간 테두리 + ★ 표시, Effect size 박스

### 옵션

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--clip-pct` | `5.0` | 컬러맵 클리핑 백분위수 (5 → P5~P95, 0이면 비활성) |
| `--outlier-n` | `3` | 양쪽 이상치 개수 (3 → 상위 3·하위 3) |
| `--out` | `experiment_1/coords_map.html` | 출력 HTML 경로 |
| `--progress` | `experiment_1/progress.json` | 진행 상태 파일 |
| `--coords` | `experiment_1/coords_korea.csv` | 좌표 CSV |
| `--results-dir` | 자동 탐지 | results/ 폴더 경로 |
| `--hist-format` | `pdf` | 히스토그램 출력 포맷 (`pdf` 또는 `png`) |

### stat.txt 구조

```
320행 = 64룰 × 5블록  (순서: Reward → Time → PDR → RewardWOG → PDRWOG)
각 행: rule_name  mean  std  95%CI_half
시각화 값 = 블록별 64개 mean의 평균 (mean of means)
```

이상치 기준: 각 블록 평균값 기준 상위/하위 N개 (지표 방향 고려: Reward은 낮은 쪽이 열악, PDR/Time은 높은 쪽이 열악).
