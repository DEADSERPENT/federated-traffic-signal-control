# ResilNet-FL IEEE Paper - Compilation Instructions

## Paper Information
- **Title**: ResilNet-FL: A Privacy-Preserving and Network-Resilient Federated Learning Framework for Intelligent Traffic Signal Control
- **Format**: IEEE Conference/Journal Paper (2-column format)
- **File**: `ResilNet_FL_IEEE_Paper.tex`
- **Length**: 10-12 pages (expanded version with theoretical analysis)
- **References**: 24 citations

## Prerequisites

### Required LaTeX Distribution
Install one of the following:
- **Windows**: [MiKTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/)
- **macOS**: [MacTeX](https://tug.org/mactex/)
- **Linux**: `sudo apt-get install texlive-full` (Ubuntu/Debian)

### Required Packages
The paper uses the following LaTeX packages (most are included in standard distributions):
- `IEEEtran` (IEEE document class)
- `cite` (citations)
- `amsmath`, `amssymb`, `amsfonts` (mathematics)
- `algorithmic`, `algorithm` (algorithms)
- `graphicx` (figures)
- `textcomp`, `xcolor` (text formatting)
- `hyperref` (hyperlinks)
- `booktabs`, `multirow` (tables)
- `subcaption` (subfigures)
- `balance` (column balancing)

## Compilation Methods

### Method 1: Command Line (Recommended)

```bash
# Navigate to paper directory
cd paper

# Compile with pdflatex (run twice for references)
pdflatex ResilNet_FL_IEEE_Paper.tex
pdflatex ResilNet_FL_IEEE_Paper.tex
```

### Method 2: Using latexmk (Automatic)

```bash
cd paper
latexmk -pdf ResilNet_FL_IEEE_Paper.tex
```

### Method 3: Overleaf (Online)
1. Create a new project on [Overleaf](https://www.overleaf.com/)
2. Upload `ResilNet_FL_IEEE_Paper.tex`
3. Upload all figures from `../results/ieee/`:
   - `system_architecture.png`
   - `ieee_fl_convergence.png`
   - `ieee_method_comparison.png`
   - `ieee_network_stress.png`
   - `ieee_generalization.png`
   - `ieee_ablation_study.png`
4. Update image paths in the .tex file (remove `../results/ieee/` prefix)
5. Compile

### Method 4: TeXstudio / TeXmaker (GUI)
1. Open `ResilNet_FL_IEEE_Paper.tex` in your LaTeX editor
2. Set compiler to `pdflatex`
3. Click "Build" or press F5/F6

## Figure Locations

The paper references figures from `../results/ieee/`. Ensure these files exist:

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | `system_architecture.png` | Three-layer system architecture |
| Fig. 2 | `ieee_fl_convergence.png` | FL training convergence |
| Fig. 3 | `ieee_method_comparison.png` | Method comparison bar chart |
| Fig. 4 | `ieee_network_stress.png` | Network stress test results |
| Fig. 5 | `ieee_generalization.png` | Generalization performance |
| Fig. 6 | `ieee_ablation_study.png` | Ablation study results |

## Troubleshooting

### Missing Figures
If figures are missing, run the experiment script first:
```bash
python run_ieee_experiments.py
python generate_architecture_diagram.py
```

### Package Not Found
If MiKTeX asks to install packages, click "Yes" to allow automatic installation.

For TeX Live, run:
```bash
tlmgr install <package-name>
```

### IEEEtran Class Not Found
Download from: https://www.ctan.org/pkg/ieeetran

Or install via package manager:
```bash
# MiKTeX
mpm --install=ieeetran

# TeX Live
tlmgr install ieeetran
```

### Path Issues on Windows
If figures aren't loading, try:
1. Using forward slashes: `../results/ieee/figure.png`
2. Or absolute paths
3. Or copy figures to the `paper/` directory and update paths

## Output

After successful compilation:
- `ResilNet_FL_IEEE_Paper.pdf` - The compiled paper (10+ pages)

## Paper Structure

1. **Abstract** - Summary of ResilNet-FL contribution
2. **Introduction** - Problem statement and motivation
3. **Related Work** - Literature review
4. **System Model** - Traffic, network, and edge computing models
5. **ResilNet-FL Framework** - FedProx algorithm and architecture
6. **Experimental Setup** - Simulation configuration
7. **Evaluation** - Results and analysis
8. **Conclusion** - Summary and future work
9. **References** - 13 cited works

## Contact

For issues with compilation, please check:
1. All required packages are installed
2. Figure paths are correct
3. LaTeX distribution is up to date
