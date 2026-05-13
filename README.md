# SIDF Lab

SIDF Lab is a research workspace for Stochastic Image Description Format experiments.

Current focus:

- Python prototypes for stochastic image reconstruction.
- Edge-preserving annealing models.
- Confidence-aware multi-resolution reconstruction.
- A future Rust core for deterministic, faster decoding.

Start here:

- [Research notes](docs/sidf-research-notes.md)
- [Repository architecture](docs/repository-architecture.md)
- [Research plan](docs/research-plan.md)
- [Research state](docs/research-state.md)
- [Agent guide](AGENTS.md)

Repository workflow:

- Save generated experiment/test images under `results/<date>-<short-name>/`.
- Use GitHub Issues with `t:*` and `p:*` labels for research tasks.
- Use GitHub Projects for task status.
- Keep specs, experiments, references, and implementation separate.
- GitHub repository: `nana-nun/sidf-lab`.
- Python environment: `.venv + requirements.txt`.

Python setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

Minimal CLI check:

```powershell
$env:PYTHONPATH = "src"
python -m sidf_lab.cli
```
