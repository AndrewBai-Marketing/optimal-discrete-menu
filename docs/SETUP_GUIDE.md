# Setup Guide for Replication

This guide helps collaborators replicate the simulation results from `simulation_results.tex`.

## Quick Start (Windows with Visual Studio)

### Prerequisites

1. **Python 3.10+** with pip
2. **Microsoft Visual Studio 2019 or 2022** (Community Edition is free)
   - Download: https://visualstudio.microsoft.com/downloads/
   - During installation, select "Desktop development with C++"
3. **Git** (optional, for version control)

### Step 1: Install Python Dependencies

Open Command Prompt or PowerShell and navigate to the project directory:

```bash
cd path\to\optimaldiscretemenu
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install numpy pandas matplotlib scipy
```

### Step 2: Compile C++ Library

**Option A: Use the batch script (recommended)**

Simply double-click `COMPILE_NOW.bat` or run from command prompt:

```bash
COMPILE_NOW.bat
```

**Option B: Manual compilation**

Open **x64 Native Tools Command Prompt for VS 2022** (search in Start menu), then:

```bash
cd path\to\optimaldiscretemenu
cl /O2 /openmp /EHsc /LD /std:c++17 /I"C:\eigen-3.4.1\eigen-3.4.1" ot_core.cpp /link /OUT:ot_core.dll
```

**Note**: You'll need Eigen library (header-only). See "Installing Eigen" section below.

### Step 3: Verify Installation

Run the quick test:

```bash
python test_openmp.py
```

Expected output:
```
=== MULTISTART VERSION: PARALLEL WITH OPENMP (16 threads) ===
```

If you see this, you're ready to go!

### Step 4: Run Simulations

To replicate the main results:

```bash
# Homogeneous baseline (20k consumers, ~10 seconds)
python test_homogeneous_baseline_proper.py

# Premium niche scenario (50k consumers, ~30 seconds)
python test_early_adopters_proper.py

# All 5 realistic scenarios (~5 minutes total)
python test_realistic_scenarios_proper.py

# Computational benchmark (5k to 1M, ~20 minutes)
python benchmark_computation_time_extended.py
```

## Installing Eigen (Required for C++ compilation)

Eigen is a header-only library (no compilation needed).

### Windows:

1. Download Eigen from: https://eigen.tuxfamily.org/index.php?title=Main_Page
2. Extract to `C:\eigen-3.4.1\` (or any location)
3. Update `COMPILE_NOW.bat` to point to your Eigen directory:
   ```batch
   /I"C:\your\path\to\eigen-3.4.1"
   ```

### Alternative: Use pre-compiled DLL

If you can't compile the C++ code, you can use the pre-compiled `ot_core.dll` included in the Dropbox folder. Just make sure it's in the same directory as the Python scripts.

**Important**: The DLL is compiled for Windows x64. Mac/Linux users will need to compile from source (see below).

## Cross-Platform Setup

### macOS

1. Install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```

2. Install Eigen via Homebrew:
   ```bash
   brew install eigen
   ```

3. Compile the C++ library:
   ```bash
   clang++ -O3 -Xpreprocessor -fopenmp -std=c++17 -shared -fPIC \
           -I/opt/homebrew/include/eigen3 \
           ot_core.cpp -o ot_core.dylib -lomp
   ```

4. Update Python scripts to load `ot_core.dylib` instead of `ot_core.dll`

### Linux (Ubuntu/Debian)

1. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential libeigen3-dev libomp-dev
   ```

2. Compile the C++ library:
   ```bash
   g++ -O3 -fopenmp -std=c++17 -shared -fPIC \
       -I/usr/include/eigen3 \
       ot_core.cpp -o ot_core.so
   ```

3. Update Python scripts to load `ot_core.so` instead of `ot_core.dll`

## Troubleshooting

### "ot_core.dll not found"

- Make sure you compiled the DLL successfully
- Check that `ot_core.dll` is in the same directory as your Python scripts
- On Windows, the DLL might be blocked by antivirus - add an exception

### "Cannot find Eigen headers"

- Verify Eigen is installed at the path specified in compilation command
- Update the `/I"path\to\eigen"` flag to match your installation

### "OpenMP not working" / "Sequential execution"

- Ensure you used `/openmp` flag (Windows) or `-fopenmp` (Mac/Linux)
- Check compilation output for OpenMP-related warnings
- Run `test_openmp.py` to verify parallel execution

### Python Import Errors

```bash
pip install --upgrade numpy pandas matplotlib scipy
```

### Performance Issues

- Verify OpenMP is working: `python test_openmp.py`
- Check number of threads being used (should match CPU cores)
- For large datasets (>100k), increase timeout in Python scripts

## File Structure

```
optimaldiscretemenu/
├── SETUP_GUIDE.md              # This file
├── requirements.txt            # Python dependencies
├── COMPILE_NOW.bat             # Windows compilation script
├── ot_core.cpp                 # C++ implementation
├── ot_core.dll                 # Pre-compiled Windows DLL
├── ks_from_sim.py              # Kohli-Sukumar baseline
├── simulation_results.tex      # Main results document
├── test_*.py                   # Test scripts for each scenario
├── benchmark_*.py              # Computational benchmarks
└── scenario_*.csv              # Generated datasets
```

## Replication Workflow

For a complete replication of the paper results:

1. **Compile C++ library** (Step 2 above)
2. **Generate scenarios** (optional, CSVs already included):
   ```bash
   python create_realistic_scenarios.py
   python create_homogeneous_baseline.py
   python create_early_adopters.py
   ```
3. **Run experiments**:
   ```bash
   python test_realistic_scenarios_proper.py
   python test_homogeneous_baseline_proper.py
   python test_early_adopters_proper.py
   python benchmark_computation_time_extended.py
   ```
4. **Results** are saved to CSV files and printed to console

## Expected Runtime

- **Homogeneous baseline** (20k): ~10 seconds
- **Early adopters** (20k): ~15 seconds
- **5 realistic scenarios** (20k each): ~5 minutes total
- **Computational benchmark** (5k to 1M): ~20 minutes total

All on a modern laptop with 16 logical cores.

## Contact

If you encounter issues not covered in this guide:
1. Check that all prerequisites are installed
2. Verify the C++ compilation succeeded (you should see `ot_core.dll`)
3. Test OpenMP with `python test_openmp.py`

For questions about the methods or results, refer to `simulation_results.tex` which contains detailed explanations of all data-generating processes and algorithms.
