import numpy as np
import pandas as pd

try:
    bus_df = pd.read_csv("Bus.txt", sep=r"\s+", header=None)
    line_df = pd.read_csv("Line.txt", sep=r"\s+", header=None)
except Exception:
    bus_df = pd.read_csv("Newton-Raphson method/Bus.txt", sep=r"\s+", header=None)
    line_df = pd.read_csv("Newton-Raphson method/Line.txt", sep=r"\s+", header=None)

bus_idx = bus_df[0].values
bus_type = bus_df[1].values
pg = bus_df[2].values
qg = bus_df[3].values
pl = bus_df[4].values
ql = bus_df[5].values
v = bus_df[6].values.astype(float)
delta = np.radians(bus_df[7].values.astype(float))

p_spec = pg - pl
q_spec = qg - ql

n_bus = len(bus_df)
y_bus = np.zeros((n_bus, n_bus), dtype=complex)

for _, line in line_df.iterrows():
    from_bus = int(line[0]) - 1
    to_bus = int(line[1]) - 1
    r = line[2]
    x = line[3]
    b = line[4]
    z = complex(r, x)
    y = 1.0 / z
    y_bus[from_bus, from_bus] += y + 1j * (b / 2.0)
    y_bus[to_bus, to_bus] += y + 1j * (b / 2.0)
    y_bus[from_bus, to_bus] -= y
    y_bus[to_bus, from_bus] -= y

g = np.real(y_bus)
b = np.imag(y_bus)

pv_buses = np.where(bus_type == 2)[0]
pq_buses = np.where(bus_type == 3)[0]
pv_pq_buses = np.sort(np.concatenate([pv_buses, pq_buses]))

tol = 1e-6
max_iter = 100

for iteration in range(max_iter):
    v_complex = v * np.exp(1j * delta)
    i_complex = y_bus @ v_complex
    s_calc = v_complex * np.conj(i_complex)
    p_calc = np.real(s_calc)
    q_calc = np.imag(s_calc)
    
    dp = p_spec - p_calc
    dq = q_spec - q_calc
    
    mismatch = np.concatenate([dp[pv_pq_buses], dq[pq_buses]])
    if np.max(np.abs(mismatch)) < tol:
        break
        
    diff_delta = delta[:, None] - delta[None, :]
    v_prod = v[:, None] * v[None, :]
    v_row = v[:, None]
    
    dp_ddelta = v_prod * (g * np.sin(diff_delta) - b * np.cos(diff_delta))
    np.fill_diagonal(dp_ddelta, -q_calc - b.diagonal() * (v ** 2))
    
    dp_dv = v_row * (g * np.cos(diff_delta) + b * np.sin(diff_delta))
    np.fill_diagonal(dp_dv, p_calc / v + g.diagonal() * v)
    
    dq_ddelta = -v_prod * (g * np.cos(diff_delta) + b * np.sin(diff_delta))
    np.fill_diagonal(dq_ddelta, p_calc - g.diagonal() * (v ** 2))
    
    dq_dv = v_row * (g * np.sin(diff_delta) - b * np.cos(diff_delta))
    np.fill_diagonal(dq_dv, q_calc / v - b.diagonal() * v)
    
    j11 = dp_ddelta[np.ix_(pv_pq_buses, pv_pq_buses)]
    j12 = dp_dv[np.ix_(pv_pq_buses, pq_buses)]
    j21 = dq_ddelta[np.ix_(pq_buses, pv_pq_buses)]
    j22 = dq_dv[np.ix_(pq_buses, pq_buses)]
    
    jacobian = np.block([[j11, j12], [j21, j22]])
    dx = np.linalg.solve(jacobian, mismatch)
    
    n_pv_pq = len(pv_pq_buses)
    delta[pv_pq_buses] += dx[:n_pv_pq]
    v[pq_buses] += dx[n_pv_pq:]

v_complex = v * np.exp(1j * delta)
i_complex = y_bus @ v_complex
s_calc = v_complex * np.conj(i_complex)
p_calc = np.real(s_calc)
q_calc = np.imag(s_calc)

results = pd.DataFrame({
    "Bus": bus_idx,
    "Type": bus_type,
    "V (pu)": np.round(v, 4),
    "Angle (deg)": np.round(np.degrees(delta), 4),
    "P (pu)": np.round(p_calc, 4),
    "Q (pu)": np.round(q_calc, 4)
})

print(results.to_string(index=False))