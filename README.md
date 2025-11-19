# Optimal Discrete Menu Design: Replication Package

Sharing my Optimal Transport Code

## Quick Start (2 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Try the Interactive Challenge

Want to see how hard menu design is? Try designing your own menu and compete against the OT algorithm:

```bash
python play_menu_challenge.py
```

This interactive tool lets you design bundles and prices for real consumer data, then shows how your profit compares to the OT solution. Great for building intuition!

**Windows & Mac users**: Pre-compiled binaries (`ot_core.dll` and `ot_core.dylib`) are included, so this works out of the box!

**Linux users**: You'll need to compile from source first (see "Compiling from Source" section below).

### Step 3: Verify Installation (Optional)

The repository includes pre-compiled binaries for Windows (x64) and macOS. Verify it works:

```bash
python test_openmp.py
```

Expected output: `=== MULTISTART VERSION: PARALLEL WITH OPENMP (16 threads) ===`

### Step 4: Run Full Simulations (Optional)

Run all four simulations from the paper:

```bash
# Simulation 1: Premium Niche Market (50k consumers)
python test_early_adopters_proper.py

# Simulations 2 & 3: Enterprise Software + Streaming Platform (20k each)
python test_enterprise_streaming_proper.py

# Simulation 4: Homogeneous Baseline (20k consumers)
python test_homogeneous_baseline_proper.py
```

Each simulation runs in ~30 seconds to 5 minutes depending on your CPU.

## Results Summary

| Simulation | Scenario | OT Advantage |
|-----------|----------|--------------|
| 1 | Premium Niche (50k) | **+27.74%** |
| 2 | Enterprise Software | **+9.43%** |
| 3 | Streaming Platform | **+9.07%** |
| 4 | Homogeneous Baseline | +1.30% |

See [docs/SIMULATION_RESULTS.md](docs/SIMULATION_RESULTS.md) or [docs/simulation_results.tex](docs/simulation_results.tex) for complete methodology and results.

## Compiling from Source (Optional)

**Note**: Pre-compiled binaries are included for Windows and macOS. Only compile from source if needed.

### Windows

Double-click `COMPILE_NOW.bat` or run from "x64 Native Tools Command Prompt for VS 2022":

```bash
COMPILE_NOW.bat
```

Requires:
- Visual Studio 2019+ with C++ tools
- Eigen 3.4+ (update path in `COMPILE_NOW.bat`)

### Linux

```bash
# Install Eigen
sudo apt-get install libeigen3-dev  # Debian/Ubuntu
# or download from https://eigen.tuxfamily.org/

# Compile with g++
g++ -O3 -fopenmp -shared -fPIC -std=c++17 \
    -I/usr/include/eigen3 \
    ot_core.cpp \
    -o ot_core.so
```

See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for detailed cross-platform instructions.

## Repository Contents

### Core Files (run simulations)

- `ot_core.cpp` / `ot_core.dll` - C++ Optimal Transport implementation with OpenMP
- `ks_from_sim.py` - Kohli-Sukumar (1990) baseline algorithm
- `utils_train_test_split.py` - 80/20 train/test splitting utilities

### Experiment Scripts

- `test_early_adopters_proper.py` - Simulation 1 (Premium Niche)
- `test_enterprise_streaming_proper.py` - Simulations 2 & 3 (Enterprise + Streaming)
- `test_homogeneous_baseline_proper.py` - Simulation 4 (Baseline)
- `benchmark_computation_time_extended.py` - Scalability analysis (5k to 1M consumers)
- `play_menu_challenge.py` - Interactive: Design your own menu, compete against OT!

### Data Generation

- `create_early_adopters.py` - DGP for Simulation 1
- `create_enterprise_streaming.py` - DGP for Simulations 2 & 3
- `create_homogeneous_baseline.py` - DGP for Simulation 4

The test scripts automatically generate data if CSV files don't exist.

### Documentation

- [docs/SIMULATION_RESULTS.md](docs/SIMULATION_RESULTS.md) - Complete results with DGPs, tables, and analysis (readable markdown)
- [docs/simulation_results.tex](docs/simulation_results.tex) - Complete results (LaTeX version)
- [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) - Detailed setup instructions for all platforms
- [docs/figures/computation_time_benchmark.pdf](docs/figures/computation_time_benchmark.pdf) - Scalability figure

## System Requirements

- **Python 3.10+** with numpy, pandas, matplotlib, scipy
- **Windows**: Pre-compiled DLL included (x64)
- **macOS**: Pre-compiled dylib included (Intel & Apple Silicon)
- **Linux**: Compile from source - requires GCC or Clang
- **Eigen 3.4+**: Required only if compiling from source (header-only library)

## Computational Benchmark

To replicate Figure 4 (computational performance):

```bash
python benchmark_computation_time_extended.py
```

Tests both algorithms on datasets from 5k to 1M consumers. Runtime: ~20 minutes.

**Key finding**: OT is 30-104× slower than KS but still highly practical:
- 20k consumers: KS = 0.14s, OT = 6s
- 1M consumers: KS = 8s, OT = 14 minutes

The 9-28% profit improvement easily justifies the modest computational overhead.

## Troubleshooting

**"Cannot load ot_core.dll"**
- Windows x64 only. Mac/Linux users must compile from source.
- Ensure DLL is in the same directory as Python scripts.

**"OpenMP not detected"**
- Verify compilation used `/openmp` (Windows) or `-fopenmp` (Mac/Linux)
- Run `python test_openmp.py` to diagnose

**Compilation errors**
- Install Eigen from https://eigen.tuxfamily.org/
- Update paths in `COMPILE_NOW.bat` (Windows) or compile command (Mac/Linux)

See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for detailed troubleshooting.

## Citation

If you use this code, please cite:

```
https://github.com/AndrewBai-Marketing/optimal-discrete-menu
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

- Setup/replication questions: See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)
- Methodology questions: See [docs/simulation_results.tex](docs/simulation_results.tex)
- Other questions: sharng@uchicago.edu

