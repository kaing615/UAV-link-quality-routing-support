# English manuscript (international submission)

English translation of `../nckh_paper`, same locked numbers. The Vietnamese
version stays authoritative for the NCKH report and the defence; this directory
is only for international submission.

Principal/corresponding author: Nguyen Dinh Tam, 23521389@gm.uit.edu.vn.
Co-author: Vo Cong Vinh. Supervisor: M.Sc. Dang Le Bao Chuong.

`main.pdf` is the current export (14 pages, includes the v10 confirmatory
section in `results_v10.tex`). Build with XeLaTeX:

```text
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Layout is generic; a venue template still has to be applied once a venue is
chosen.

## Figures

Regenerated in English from the same audited traces as the Vietnamese version
(same episode IDs, no reselection):

```text
py -I reports/nckh_paper_en/tools/make_study_design_en.py
py -I reports/nckh_paper_en/tools/plot_motion_example_en.py
```

`plot_motion_example_en.py` reads `reports/uav_service_v9/results` and the
audited selection in `../nckh_final/evidence/v9-audit.json`.

## Number formatting

All decimal separators were converted from comma to point. When checking
against `../nckh_final/evidence/`, remember that the Vietnamese sources write
`0,66` where this version writes `0.66`.

## Blocking items before submission

1. **Run v11** (`docs/service_v11_protocol.md`). v10 is done: it ran on
   2026-11-24 and is reported in `results_v10.tex` (H1 $+0.08$ pp
   [$-0.11$, 0.29], no difference detected), so the tuned-baseline question a
   reviewer would raise first is now answered in the paper. What remains is
   robustness across training seeds (v11 H1) and the parameter-matched
   GNN-versus-MLP ablation (v11 H2); both are still stated as limitations.
2. **Supervisor sign-off on the thesis change** — from "we propose a GNN" to
   "we test what a learned forecaster is actually worth".
3. **Apply the venue template** once the venue is fixed.

## Venue notes (verified November 2026)

| Venue | Cost | Fit |
|---|---|---|
| Ad Hoc Networks / Computer Communications (Elsevier) | No APC on the subscription route | Best topical fit; slow review; highest bar |
| IEEE Access | approx. USD 2,160 + tax, 20% IEEE society discount | Broad scope, fast, accepts methodology papers |
| Drones (MDPI) | CHF 2,600 | Already published Liu et al. 2025 on this topic |
| IEEE ATC (Vietnam, English, IEEE-indexed) | Registration only | Cheapest international route; good first venue |
