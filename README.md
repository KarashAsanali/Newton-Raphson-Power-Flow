# Newton-Raphson-Power-Flow

Newton-Raphson power flow calculation algorithm implemented in Python using NumPy and Pandas.

## Requirements

- Python 3.x
- `numpy`
- `pandas`

Install dependencies:
```bash
pip install numpy pandas
```

## Running the Power Flow

Run the calculation using:
```bash
python main.py
```

## Input Data Files

- `Bus.txt` / `Bus.csv`: Bus data ($S_{base} = 1\text{ MVA}, V_{base} = 12.66\text{ kV}$)
  - Column 1: Bus index number
  - Column 2: Bus type (`1` = Slack, `2` = PV, `3` = PQ)
  - Column 3: Active power generation ($P_g$ in p.u.)
  - Column 4: Reactive power generation ($Q_g$ in p.u.)
  - Column 5: Active power load ($P_l$ in p.u.)
  - Column 6: Reactive power load ($Q_l$ in p.u.)
  - Column 7: Voltage magnitude ($|V|$ in p.u.)
  - Column 8: Voltage angle ($\delta$ in degrees)

- `Line.txt` / `Line.csv`: Branch / line parameters
  - Column 1: From bus index
  - Column 2: To bus index
  - Column 3: Line resistance ($R$ in p.u.)
  - Column 4: Line reactance ($X$ in p.u.)
  - Column 5: Line susceptance ($B$ in p.u.)