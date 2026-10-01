# observability-lumo-espoo

Reproducible analysis of Sentinel-1 observability over two structures (the LUMO
lattice tower, Hannover, and the Espoo Kurttila telecom mast) plus a recomputation
of LUMO's modal frequencies and damping ratios from the raw SHM recordings.

## Folder structure

```
observability-lumo-espoo/
├── code/     analysis + plotting scripts: read data/*.json (and, for two of
│             them, raw .mat recordings or a Sentinel-1 SLC cache) and write
│             data/*.json or data/*.csv — see code/README.md
├── data/     input JSON artifacts and the CSV/JSON files the code/ scripts
│             generate from them — see data/README.md
├── figures/  one .sh wrapper per plotting script in code/, plus the .md/.png
│             reports those scripts write — see figures/README.md
├── secrets/  local credential file (lumo_slc.env) for the two network scripts
│             in code/; git-ignored, never committed — see secrets/README.md
├── .gitignore
└── README.md
```

See the per-folder `README.md` for what each script/file is for and how to run it:

* [code/README.md](code/README.md) — the ten analysis/plotting scripts, one
  paragraph each plus the commands to run them.
* [data/README.md](data/README.md) — what every input/generated file in `data/`
  is and which script reads or writes it.
* [figures/README.md](figures/README.md) — the four `.sh` wrappers that run
  the plotting scripts and write their `.md`/`.png` reports here.
* [secrets/README.md](secrets/README.md) — what the credential file is for and
  why reviewers don't need it.

