# Kế hoạch Production hóa — UAV-GNN Link Quality (AWS EKS, Full Tier 1+2+3)

> Trạng thái: **PLAN ONLY** — chưa triển khai. Tài liệu này mô tả lộ trình biến pipeline
> MLOps hiện tại thành một sản phẩm production thật trên AWS.

## 1. Mục tiêu & phạm vi

Biến hệ thống hiện tại (FastAPI serving + DVC + GitHub Actions + Kustomize/ArgoCD)
thành một sản phẩm production hoàn chỉnh trên **AWS EKS**, gồm:

- **IaC**: toàn bộ hạ tầng dựng bằng Terraform (không bấm tay trên console).
- **GitOps**: ArgoCD là nguồn sự thật duy nhất cho mọi thứ chạy trong cluster.
- **Observability 3 trụ**: Metrics (Prometheus/Grafana), Logs (Loki), Traces (Tempo/OTel).
- **ML Ops thật**: MLflow (tracking + registry), Evidently (drift), retrain orchestration.
- **Production hardening**: Ingress+TLS, autoscaling, alerting/SLO, secrets, image signing,
  WAF, backup/DR, cost & HA.

Phạm vi = **Full Tier 1 + 2 + 3** + các gap bổ sung (GPU, prediction store, backup/DR,
cost/HA, orchestration, runbooks).

## 2. Hiện trạng (điểm xuất phát)

| Có rồi | Còn thiếu cho production |
|---|---|
| DVC (data versioning) | Model registry / experiment tracking UI |
| Dockerfile + Dockerfile.serve | Image scan / sign / SBOM, container hardening |
| GitHub Actions (ci, build-and-push) | Provisioning hạ tầng (IaC), promotion gate |
| Kustomize base + overlay staging/prod | Ingress + TLS + domain |
| ArgoCD Application (staging) | Autoscaling (HPA/KEDA), HA (PDB/spread) |
| FastAPI `/health`, `/predict` | `/metrics`, logging có cấu trúc, tracing, auth/WAF |
| Data validation (`src/validation`) | Drift monitoring online + prediction store |
| | Alerting/SLO, secrets, backup/DR, cost monitoring |

**Lưu ý code:** `src/serving/app.py` hiện **chưa expose `/metrics`** và dùng global model
state load 1 lần lúc startup → cần thêm instrumentation; service hiện là `ClusterIP`.

**Lưu ý GPU:** đây là mô hình **GNN** → train (và có thể inference) cần GPU; hạ tầng phải
có node group GPU, không chỉ CPU.

## 3. Kiến trúc đích (AWS)

```
                         ┌────────────────────────────────────────────────┐
 Terraform ─ provision ─▶│  VPC · EKS(CPU+GPU node) · ECR · S3 · RDS · R53 │
                         │  IAM/IRSA · ALB Controller · WAF · NAT          │
                         └────────────────────────────────────────────────┘
                                          │ bootstrap
                                          ▼
 GitHub Actions ─build+test+Trivy+cosign+SBOM─▶ ECR
   │ (staging→smoke test→approve→prod)
                                          │
                                  ArgoCD (app-of-apps, GitOps)
                                          │ sync
        ┌─────────────────────────────────┼────────────────────────────────┐
        ▼                                  ▼                                 ▼
  Platform add-ons                    Application                      ML platform
  • cert-manager                  WAF→ALB Ingress(TLS, R53)            • MLflow (RDS+S3)
  • external-dns                       │                               • Evidently CronJob
  • kube-prometheus-stack        FastAPI /predict /metrics             • Argo Workflows
  • loki + promtail                    │  ▲ HPA/KEDA (QPS,p95)            (retrain/ingest)
  • tempo + otel-collector       OTel SDK → traces                     • prediction store
  • metrics-server / KEDA        async log preds → S3/Athena             (S3+Athena)
  • sealed-secrets / ESO         PDB + topologySpread (HA)
  • velero (backup)
  • kubecost (cost)

  Prometheus ─▶ Grafana(metrics/SLO) ─▶ Alertmanager ─▶ Slack/Discord
  Loki ───────▶ Grafana(logs)        blackbox-exporter → /health từ ngoài
  Tempo ──────▶ Grafana(traces)
```

**Quyết định kiến trúc chính:**
1. **Bootstrap pattern**: Terraform chỉ dựng *infra + ArgoCD*. Mọi add-on & app cài qua
   ArgoCD **app-of-apps** → tránh trộn lifecycle Terraform/Helm, giữ GitOps thuần.
2. **IRSA** cho mọi pod cần quyền AWS (ALB controller, external-dns, cert-manager DNS-01,
   pod đọc S3/MLflow, prediction-store writer). Không dùng access key tĩnh.
3. **MLflow backend = RDS Postgres (Multi-AZ)**, artifact store = S3 (dùng chung bucket với
   DVC remote, khác prefix).
4. **TLS**: cert-manager + Let's Encrypt (DNS-01 qua Route53) — domain thật, auto-renew.
5. **GPU**: managed node group riêng (g4dn) có taint; train/inference GPU lên đó qua toleration.
   Node CPU cho serving nhẹ + platform.
6. **Prediction store**: mỗi request `/predict` ghi async input+output ra S3 (query bằng
   Athena) — là **nguồn dữ liệu cho drift monitoring** và audit.
7. **Serving nâng cao (Tier 3)**: cân nhắc **KServe** (canary/A-B + scale-to-zero); nếu rủi ro
   cao thì giữ FastAPI + **Argo Rollouts** cho canary.

## 4. Cấu trúc thư mục bổ sung (đề xuất)

```
terraform/
  modules/            # vpc, eks(cpu+gpu), ecr, rds, iam-irsa, s3, waf, athena
  envs/
    staging/          # backend.tf (S3+DynamoDB lock), main.tf, variables.tf
    production/
deploy/
  argocd/
    root-app.yaml             # app-of-apps
    apps/                     # 1 Application / 1 add-on hoặc app
      platform-*.yaml         # alb, external-dns, cert-manager, velero, kubecost...
      mlflow.yaml
      monitoring.yaml
      argo-workflows.yaml
  kustomize/
    base/
      ingress.yaml            # MỚI (ALB + WAF assoc)
      servicemonitor.yaml     # MỚI
      hpa.yaml / scaledobject.yaml  # MỚI
      prometheus-rules.yaml   # MỚI (SLO burn-rate)
      pdb.yaml                # MỚI (HA)
  helm-values/                # values override cho các chart
observability/
  grafana-dashboards/         # JSON: api, ml, infra, slo, cost
monitoring/
  evidently/                  # script drift + Dockerfile job
orchestration/
  argo-workflows/             # template retrain / data-ingest
load-test/
  k6/                         # script k6
docs/
  runbooks/                   # incident, rollback-model, restore-db, dr-drill
.github/
  renovate.json / dependabot.yml   # auto dependency PR
```

## 5. Lộ trình theo giai đoạn (milestones)

> Mỗi phase có Definition of Done (DoD) rõ ràng, làm tuần tự, mỗi phase commit/PR riêng.

### Phase 0 — Nền tảng Terraform & remote state
- Backend Terraform: S3 + DynamoDB lock.
- Module `vpc` (public/private subnet, NAT), `iam`.
- **DoD**: `terraform plan/apply` chạy được, state lưu trên S3.

### Phase 1 — EKS + registry + storage (CPU **và** GPU)
- Module `eks`: managed node group **CPU** (serving/platform) + node group **GPU** (g4dn, có
  taint) + NVIDIA device plugin; addons vpc-cni, coredns, kube-proxy, ebs-csi.
- ECR repos (train, serve). S3 bucket (DVC remote + MLflow artifacts + **prediction store**,
  bật versioning). **Athena** + Glue table cho prediction store.
- RDS Postgres **Multi-AZ** (MLflow). IRSA roles: ALB controller, external-dns, cert-manager,
  app-s3, prediction-writer.
- **DoD**: `kubectl get nodes` thấy cả CPU+GPU; pod GPU test chạy `nvidia-smi`; ECR/S3/RDS/Athena tồn tại; DVC remote trỏ S3 mới.

### Phase 2 — GitOps bootstrap + platform add-ons
- Cài ArgoCD (Terraform Helm provider, một lần) → root app-of-apps.
- ArgoCD apps: AWS LB Controller, external-dns, cert-manager, metrics-server, sealed-secrets/ESO.
- **DoD**: ArgoCD UI xanh; tạo Ingress test ra ALB + DNS + cert tự động.

### Phase 3 — Đưa app lên kèm Ingress + TLS + promotion gate
- Thêm `ingress.yaml` (ALB, host `api.<domain>`), chuyển traffic vào service.
- Overlay staging/production khác host + replicas.
- **CI promotion gate**: GH Actions deploy staging → chạy **smoke test** tự động (`/health`,
  1 `/predict` mẫu) → **manual approval** (GH environment) → prod.
- **DoD**: `https://api-staging.<domain>/health` 200 với cert hợp lệ; pipeline staging→prod có cổng duyệt + smoke test chặn được bản lỗi.

### Phase 4 — Metrics, SLO & Prediction store (Tier 1)
- **Code (`app.py`)**: thêm `/metrics` (`prometheus-fastapi-instrumentator`) + custom ML
  metrics (số edge/req, histogram `stability_score`, tỉ lệ `stable`, label `model_id`,
  inference latency); **ghi async prediction (input+output) ra S3** (prediction store).
- Cài `kube-prometheus-stack` qua ArgoCD; `ServiceMonitor` scrape `/metrics`.
- Grafana dashboards: API, ML, Infra. **SLO**: định nghĩa SLI (p95 latency, availability,
  error rate) + `PrometheusRule` **multi-window burn-rate** + panel error budget.
- Alertmanager → Slack/Discord (error rate, burn-rate, pod down, no-traffic, model not loaded).
- **DoD**: Grafana hiển thị QPS/latency/score thật + error budget; bắn được 1 alert test; prediction xuất hiện trong S3/Athena.

### Phase 5 — Logging & Tracing (Tier 1/3)
- **Code**: logging JSON có cấu trúc (request_id) + OpenTelemetry SDK trong FastAPI.
- Loki + Promtail (logs) → Grafana; otel-collector + Tempo (traces) → Grafana.
- **DoD**: 1 request `/predict` truy được cả log + trace span trong Grafana.

### Phase 6 — Autoscaling, HA & Load test (Tier 1/2)
- HPA (CPU) → nâng lên **KEDA**/prometheus-adapter scale theo QPS/p95 latency.
- **HA**: `PodDisruptionBudget` + `topologySpreadConstraints` (trải pod qua AZ).
- `load-test/k6` kịch bản tải tăng dần.
- **DoD**: chạy k6 thấy pod scale 1→N rồi co lại; drain 1 node không gây downtime (PDB hoạt động).

### Phase 7 — MLflow (Tier 2)
- MLflow server (RDS + S3) qua ArgoCD; tích hợp tracking vào `src/training/*`.
- Đăng ký model vào Registry; `scripts/mlops/promote_model.sh` đọc registry (Staging→Prod);
  liên kết với `deploy/serving_model.json`.
- **DoD**: 1 lần train log lên MLflow UI; promote model → serve đọc đúng version.

### Phase 8 — Drift monitoring & retrain orchestration (Tier 2)
- CronJob **Evidently**: đọc **prediction store (Phase 4)** so với reference dataset → đẩy
  metric drift vào Prometheus (pushgateway) → alert khi vượt ngưỡng.
- **Argo Workflows**: template retrain (chạy trên GPU node) + data-ingest theo lịch/sự kiện;
  drift cao → trigger retrain → log MLflow → promote.
- **DoD**: Grafana có panel drift; vượt ngưỡng → alert; trigger 1 retrain workflow end-to-end.

### Phase 9 — Security hardening (Tier 3)
- CI: **Trivy** scan image (fail nếu CRITICAL), **cosign** ký image, **syft** SBOM.
- **Container hardening**: image serve distroless/non-root, read-only rootfs, drop capabilities.
- **API hardening**: API-key/JWT auth, rate-limit (slowapi), giới hạn payload + CORS;
  **AWS WAF** gắn ALB (rate-based + managed rules).
- Secrets: Sealed Secrets / External Secrets ↔ AWS Secrets Manager (bỏ giá trị nhạy cảm khỏi
  configmap). NetworkPolicy, PodSecurity baseline/restricted.
- **DoD**: image ECR verify được chữ ký; không còn secret plaintext trong Git; WAF chặn được request quá ngưỡng; `/predict` yêu cầu auth.

### Phase 10 — Serving nâng cao (Tier 3, optional)
- KServe `InferenceService` hoặc Argo Rollouts canary (10%→50%→100%) cho model mới.
- **DoD**: deploy model mới dạng canary, rollback tự động nếu error rate tăng.

### Phase 11 — Backup & Disaster Recovery (Tier 3)
- **Velero** backup K8s resources + PV → S3 (lịch định kỳ).
- RDS automated snapshot + S3 versioning (đã bật) + cross-region copy (optional).
- Runbook restore; **DR drill** thử khôi phục.
- **DoD**: xóa thử 1 namespace rồi restore bằng Velero thành công; có snapshot RDS khôi phục được.

### Phase 12 — Cost & FinOps (Tier 3)
- **Kubecost** (hoặc OpenCost) + AWS Budgets alert; dashboard cost trong Grafana.
- Tối ưu: spot cho GPU/train, scale-to-zero serving (KServe), right-size requests/limits.
- **DoD**: dashboard cost theo namespace/workload; budget alert bắn khi vượt ngưỡng.

### Phase 13 — Vận hành: runbooks & uptime (Tier 3)
- `docs/runbooks/`: incident response, rollback model, restore DB, DR drill.
- **blackbox-exporter** ping `/health` từ ngoài (external uptime) → alert.
- **Renovate/Dependabot**: tự động PR cập nhật dependency + base image.
- **DoD**: có đủ runbook; uptime check hoạt động; 1 PR Renovate tự mở.

## 6. Thay đổi code dự kiến (tóm tắt)

| File | Thay đổi |
|---|---|
| `requirements-serve.txt` | + `prometheus-fastapi-instrumentator`, `opentelemetry-*`, `python-json-logger`, `slowapi`, `boto3` |
| `src/serving/app.py` | `/metrics`, custom ML metrics, JSON logging, OTel, **async prediction logging → S3**, API-key auth + rate limit + CORS/size limit |
| `requirements.txt` / `src/training/*` | + `mlflow`; log params/metrics/model; chạy được trên GPU node |
| `Dockerfile.serve` | distroless/non-root, read-only rootfs, drop capabilities |
| `.github/workflows/build-and-push.yml` | + Trivy + cosign + SBOM, push ECR |
| `.github/workflows/*` | promotion gate: staging → smoke test → approval → prod |
| `scripts/mlops/promote_model.sh` | đọc/ghi MLflow Model Registry |
| `monitoring/evidently/` | script + Dockerfile cho CronJob drift (đọc prediction store) |
| `orchestration/argo-workflows/` | template retrain (GPU) + data-ingest |
| `.github/renovate.json` | cấu hình auto update |

## 7. Ước tính chi phí AWS (tham khảo, on-demand, ~us-east-1)

| Resource | Cấu hình tối thiểu demo | Ước tính/tháng |
|---|---|---|
| EKS control plane | 1 cluster | ~$73 |
| Node group CPU | 2× t3.large (hoặc spot) | ~$120 (giảm mạnh nếu spot) |
| Node group GPU | 1× g4dn.xlarge (chỉ bật khi train) | ~$380 nếu chạy 24/7 — **dùng spot + scale-to-zero** |
| NAT Gateway | 1 | ~$32 + data |
| RDS Postgres (MLflow) | db.t3.micro (Multi-AZ ~×2) | ~$15–30 |
| ALB + WAF | 1 | ~$18 + LCU + ~$6 WAF |
| S3 / ECR / Route53 / Athena | nhẹ | ~$5–10 |
| **Tổng (GPU scale-to-zero)** | | **~$280–320/tháng** |

> Giảm chi phí cho đồ án: **GPU node scale-to-zero** (chỉ bật khi train/demo), **spot** cho
> GPU+train, **1 NAT** (hoặc public subnet no-NAT cho demo), RDS single-AZ khi không demo HA,
> `terraform destroy`/scale node = 0 khi nghỉ. Kubecost giúp theo dõi để cắt. Cân nhắc AWS credits học thuật.

## 8. Rủi ro & lưu ý

- **Chi phí GPU**: g4dn đắt nhất — bắt buộc scale-to-zero/spot, không để chạy 24/7.
- **Độ phức tạp IRSA/DNS/TLS/WAF**: dễ sai ở DNS-01 và IAM trust policy → làm Phase 2 cẩn thận.
- **Chicken-egg ArgoCD**: bootstrap ArgoCD bằng Terraform một lần, sau đó tự quản qua GitOps.
- **Prediction store là dependency ẩn**: phải có từ Phase 4 thì Phase 8 (drift) mới chạy.
- **Model artifact lớn**: GNN weights — đảm bảo S3/ECR layer cache hợp lý, image serve gọn.
- **Scope creep**: Tier 3 (KServe, tracing, DR, cost) là optional — ưu tiên 1→2 trước.

## 9. Thứ tự ưu tiên đề xuất

1. **Phase 0–3** (infra + GPU + app online có TLS + promotion gate) — xương sống.
2. **Phase 4** (metrics/SLO/Grafana + prediction store) — giá trị "show" cao nhất, rẻ, và mở khóa drift.
3. **Phase 7** (MLflow) — biến thành "MLOps" thực thụ.
4. **Phase 6** (autoscale + HA + k6), **Phase 5** (logs/traces).
5. **Phase 8** (drift + retrain orchestration).
6. **Phase 9** (security), **Phase 11–13** (DR, cost, vận hành), **Phase 10** (KServe canary) — hoàn thiện.
