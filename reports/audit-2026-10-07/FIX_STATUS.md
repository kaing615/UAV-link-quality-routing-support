# Trang thai sua audit 07/10/2026

Da sua ma nguon cho 11 phat hien. Bang va figure nghien cuu cu chua duoc sinh lai: checkout hien tai khong co `data/`, `outputs/` hay checkpoint thuc nghiem. Khong dung ket qua cu de ket luan cac mo hinh da duoc so sanh dung.

## Doi chieu

| Phat hien | Thay doi | Kiem chung |
| --- | --- | --- |
| 1. LORO ro ri seed | Loai tat ca scenario cung seed khoi train; manifest ghi run va nhom seed train/test. Cross-mobility cung kiem tra tach seed. | Regression chup command va kiem tra resume. |
| 2. Don vi thong ke sai | Lay trung binh scenario trong seed truoc mean/std, bootstrap va sign-flip. Giu `n_runs`/`n_pairs`, them so don vi doc lap. | Regression cohort khong can bang va p-value theo seed. |
| 3. Artifact LORO khac mau | Cache phai khop data/code/config/fold; kiem tra ca `(time,src,dst,label)` va so mau giua model. Cache cu khong co manifest se chay lai. | Regression cung so mau nhung sai danh tinh canh. Artifact thuc te can sinh lai. |
| 4. Binary ns-3 cu | Local CMake incremental; image kem checksum source/CMake/binary, sai checksum thi dung. Closed-loop truc tiep cung xac minh checksum. Dataset luu checksum binary/source. DVC theo doi them shell/CMake. | Build ns-3 3.48, regression checksum va hai lan incremental bang CMake stub. |
| 5. Serving ep Edge-SAGE | Bo ENV MODEL_ID trong image; smoke check load checkpoint staged. | Load checkpoint GAT, GraphSAGE, no-edge va decoder-only tu metadata. |
| 6. Mat Persistence | Benchmark mac dinh gom Persistence tren dung cohort; bo luot danh gia Persistence rieng trung lap. Bang LaTeX cung giu Persistence. | Regression danh sach model va full artifact suite. |
| 7. Gom cohort cu | Stage 6 va routing chi doc cac output cua invocation; closed-loop nhan `--cohort-runs` cho batch. | Regression co file lich su ngoai cohort. |
| 8. Cache inference sai | Fingerprint target/horizon/threads/warmups/repeats, checkpoint, input va code; routing/closed-loop cung fingerprint. | Regression thay threads, checkpoint va target/horizon. |
| 9. Horizon bi rut | Replay va route plan chi dung cua so co du tat ca snapshot tu t den t+H. Khong co cua so hop le thi bao loi. | Regression H=5 voi trace chi co t=0,1. |
| 10. No-route gui truc tiep | Routing planned kiem soat unicast den dich, drop khi node khong co next hop trong plan, khong fallback subnet/OLSR. PDR tinh tren offered packets. | Chay ns-3 3 UAV: no-route, direct, bat buoc relay va mat route nua cua so. |
| 11. Numeric API | Tu choi non-finite, vuot float32, loss ngoai [0,1] va cac dac trung vat ly am. | HTTP 422 voi payload loi. |

## Ket qua kiem tra

- 95/95 pytest tren Windows/Python 3.12.14 pass; hai test Bash lifecycle chay rieng tren Linux va pass.
- 22/22 script Bash qua `bash -n`; checksum/build lifecycle check pass.
- ns-3 3.48 bien dich thanh cong; bon truong hop packet delivery/forwarding thuc te pass.
- Ruff check `src scripts tests` pass; format check cac file Python da sua pass; `git diff --check` pass.
- Chua build full image nghien cuu/serving, train 400 controlled datasets hoac deploy cloud.

## Chay lai

### Kiem tra sau khi gop origin/main d07db1f

- Giu controlled scenario extensions va temporal GNN benchmark; nhom seed nhan dien them `partition_ctrl`.
- Closed-loop chuyen tiep scenario mode va background traffic tu metadata goc.
- 143 pytest pass tren Windows; 3 Bash test va 3 ns-3 binary test chay rieng tren Linux deu pass (`check_merged_pipeline.py`). Hai test serving lifecycle cung pass tren Linux.
- Rebuild scenario ns-3 3.48; route/drop/relay/PDR checks, checksum/incremental build va 22 Bash syntax checks pass.
- Test throughput dung gioi han sai so tu do chinh xac 6 chu so thap phan cua CSV.
- Ruff check va format cho cac file sua khi gop pass. Chua sinh lai artifact nghien cuu.

Tu repo root, voi ns-3 local da cai hoac research image vua rebuild:

```bash
bash scripts/run_controlled_paper_pipeline.sh
```

Pipeline se tu loai cache cu thieu manifest hoac khac fingerprint; cac bang/figure duoc tao lai o `reports/controlled_paper/`. Can giu cung image, dataset va manifest khi so sanh ket qua. LORO giu rieng output tung condition de tuong thich layout hien tai; cac condition cung seed deu loai cung nhom khoi train.

De chay lai kiem tra ns-3 doc lap, khong train mo hinh:

```powershell
rtk proxy docker build -f reports/audit-2026-10-07/ns3-check.Dockerfile -t uav-ns3-audit:local simulation/ns3
rtk proxy docker run --rm --network none --mount 'type=bind,source=C:\Study\DACN\UAV-link-quality-routing-support,target=/repo,readonly' -w /repo uav-ns3-audit:local python3 reports/audit-2026-10-07/check_ns3_routes.py /scenario/build/uav-olsr-dataset
```

`AUDIT.md` va `probe_reaudit.py` giu nguyen bang chung lich su. Probe cu assert hanh vi loi; dung `tests/test_audit_regressions.py` de kiem tra hanh vi da sua.
