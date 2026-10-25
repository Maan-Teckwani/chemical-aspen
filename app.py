"""
GreenH2 Studio — single-file Streamlit MVP
Run: streamlit run app.py
Install: pip install streamlit google-genai pandas plotly graphviz
Optional: add GEMINI_API_KEY to .streamlit/secrets.toml or your environment.
"""

import os
import json
import math
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai

st.set_page_config(
    page_title="GreenH2 Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e7edf7; }
    [data-testid="stSidebar"] { background: #101a2b; }
    [data-testid="stMetric"] {
        background: #142238; border: 1px solid #263951;
        padding: 14px 16px; border-radius: 12px;
    }
    .hero {
        padding: 26px 28px; border-radius: 18px; margin-bottom: 20px;
        background: linear-gradient(120deg, #102f35, #142238 65%, #1a2440);
        border: 1px solid #28534f;
    }
    .muted { color: #9eafc7; }
    .pill {
        display:inline-block; padding:5px 10px; border-radius:999px;
        background:#173c35; color:#8de0ba; font-size:12px; margin-right:6px;
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- Gemini helpers ----------
def get_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")

def gemini_text(prompt, model="gemini-2.5-flash"):
    key = get_api_key()
    if not key:
        raise ValueError("Add GEMINI_API_KEY in Streamlit secrets or as an environment variable.")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()

def extract_json(text):
    """Parse JSON even if the model wraps it in a Markdown code fence."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Gemini did not return a JSON object.")
    return json.loads(text[start:end + 1])

# ---------- Transparent first-pass engineering calculations ----------
def calculate_design(capacity_kg_day, specific_energy, uptime, water_factor,
                     cooling_fraction, outlet_bar, inlet_bar, compressor_eff,
                     electricity_price, water_price, capex_per_kw, capex_per_kg_day,
                     capex_other, fixed_opex_pct, stack_life, project_years,
                     discount_rate):
    # 24 h/day nameplate basis; uptime is an operating fraction.
    h2_daily = capacity_kg_day * uptime
    h2_hourly = h2_daily / 24
    power_kw = capacity_kg_day * specific_energy / 24
    electricity_kwh_day = capacity_kg_day * specific_energy * uptime
    water_kg_day = h2_daily * 9 * water_factor
    water_m3_day = water_kg_day / 1000

    # Cooling duty is a user-selected fraction of electrolyzer electricity input.
    cooling_kw = power_kw * cooling_fraction
    cooling_kwh_th_day = cooling_kw * 24 * uptime

    # Approximate isothermal compression: W = nRT ln(P2/P1) / efficiency.
    # Assumes hydrogen at 298.15 K and ideal-gas behavior; real systems need detailed design.
    r = 8.314462618
    molar_mass_kg_mol = 0.00201588
    mol_per_kg = 1 / molar_mass_kg_mol
    if outlet_bar > inlet_bar > 0:
        compressor_kwh_kg = (
            mol_per_kg * r * 298.15 * math.log(outlet_bar / inlet_bar)
            / 3_600_000 / max(compressor_eff, 0.05)
        )
    else:
        compressor_kwh_kg = 0.0
    compressor_kw = compressor_kwh_kg * h2_hourly
    compressor_kwh_day = compressor_kwh_kg * h2_daily

    annual_h2 = h2_daily * 365
    annual_electricity = (electricity_kwh_day + compressor_kwh_day) * 365
    annual_water = water_m3_day * 365
    annual_electricity_cost = annual_electricity * electricity_price
    annual_water_cost = annual_water * water_price

    electrolyzer_capex = power_kw * capex_per_kw
    compression_capex = compressor_kw * capex_per_kw * 1.5
    water_capex = water_m3_day * capex_per_kg_day
    total_capex = electrolyzer_capex + compression_capex + water_capex + capex_other
    annual_fixed_opex = total_capex * fixed_opex_pct
    annual_opex = annual_electricity_cost + annual_water_cost + annual_fixed_opex

    # Simplified annualized CAPEX using a capital recovery factor.
    i = discount_rate
    n = max(int(project_years), 1)
    crf = (i * (1 + i) ** n / ((1 + i) ** n - 1)) if i > 0 else 1 / n
    annualized_capex = total_capex * crf
    lcoh = (annualized_capex + annual_opex) / annual_h2 if annual_h2 > 0 else 0

    return {
        "h2_daily": h2_daily, "h2_hourly": h2_hourly,
        "power_kw": power_kw, "electricity_kwh_day": electricity_kwh_day,
        "water_kg_day": water_kg_day, "water_m3_day": water_m3_day,
        "cooling_kw": cooling_kw, "cooling_kwh_th_day": cooling_kwh_th_day,
        "compressor_kw": compressor_kw, "compressor_kwh_kg": compressor_kwh_kg,
        "compressor_kwh_day": compressor_kwh_day, "annual_h2": annual_h2,
        "annual_electricity": annual_electricity, "annual_water": annual_water,
        "annual_electricity_cost": annual_electricity_cost,
        "annual_water_cost": annual_water_cost, "electrolyzer_capex": electrolyzer_capex,
        "compression_capex": compression_capex, "water_capex": water_capex,
        "total_capex": total_capex, "annual_fixed_opex": annual_fixed_opex,
        "annual_opex": annual_opex, "annualized_capex": annualized_capex, "lcoh": lcoh,
    }

def default_pfd():
    return {
        "title": "Green hydrogen production — conceptual PFD",
        "nodes": [
            {"id": "water", "label": "DI water\nstorage"},
            {"id": "treat", "label": "Water\npolishing"},
            {"id": "ely", "label": "Electrolyzer\nstack"},
            {"id": "sep", "label": "Gas-liquid\nseparator"},
            {"id": "dry", "label": "H₂ drying &\npurification"},
            {"id": "comp", "label": "H₂ compressor"},
            {"id": "store", "label": "H₂ storage"},
            {"id": "o2", "label": "O₂ by-product"},
            {"id": "cool", "label": "Cooling loop"},
            {"id": "power", "label": "Power supply"},
        ],
        "edges": [
            {"from": "water", "to": "treat", "label": "DI water"},
            {"from": "treat", "to": "ely", "label": "feed water"},
            {"from": "ely", "to": "sep", "label": "H₂ + O₂ + water"},
            {"from": "sep", "to": "dry", "label": "H₂-rich gas"},
            {"from": "dry", "to": "comp", "label": "dry H₂"},
            {"from": "comp", "to": "store", "label": "compressed H₂"},
            {"from": "sep", "to": "o2", "label": "O₂ stream"},
            {"from": "cool", "to": "ely", "label": "heat removal"},
            {"from": "power", "to": "ely", "label": "DC power"},
        ],
    }

def pfd_to_dot(spec):
    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    if not nodes or not edges:
        raise ValueError("PFD needs at least one node and edge.")
    ids = {str(n.get("id", "")) for n in nodes}
    lines = [
        'digraph GreenH2 {',
        'rankdir=LR; bgcolor="transparent"; pad="0.25"; nodesep="0.45"; ranksep="0.65";',
        'node [shape=box, style="rounded,filled", fillcolor="#142b3d", color="#38b2a0", '
        'fontcolor="#e7edf7", fontname="Arial", fontsize=10, margin="0.16,0.10"];',
        'edge [color="#6ba9a0", fontcolor="#a7c9c5", fontname="Arial", fontsize=8, arrowsize=0.7];',
    ]
    for n in nodes:
        nid = str(n.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_ -]{1,40}", nid):
            continue
        label = str(n.get("label", nid)).replace("\\", "\\\\").replace('"', '\\"')
        label = label.replace("\n", "\\n")[:100]
        lines.append(f'"{nid}" [label="{label}"];')
    for e in edges:
        a, b = str(e.get("from", "")), str(e.get("to", ""))
        if a in ids and b in ids and re.fullmatch(r"[A-Za-z0-9_ -]{1,40}", a) and re.fullmatch(r"[A-Za-z0-9_ -]{1,40}", b):
            label = str(e.get("label", "")).replace('"', '\\"')[:80]
            lines.append(f'"{a}" -> "{b}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)

# ---------- Sidebar inputs ----------
st.sidebar.markdown("## ⚡ GreenH2 Studio")
st.sidebar.caption("Conceptual engineering workspace")
api_key_present = bool(get_api_key())
if api_key_present:
    st.sidebar.success("Gemini API key detected")
else:
    st.sidebar.warning("Gemini key not configured")

st.sidebar.markdown("### Plant design basis")
capacity = st.sidebar.number_input("Nameplate H₂ capacity (kg/day)", 10, 100000, 1000, step=100)
electrolyzer = st.sidebar.selectbox("Electrolyzer technology", ["Alkaline", "PEM", "SOEC (conceptual)"])
specific_energy = st.sidebar.number_input("Electrolyzer electricity (kWh/kg H₂)", 35.0, 80.0, 52.0, step=0.5)
uptime_pct = st.sidebar.slider("Operating availability (%)", 30, 100, 90)
water_factor = st.sidebar.slider("DI water design factor (× stoichiometric)", 1.0, 2.5, 1.3, 0.1)
cooling_fraction_pct = st.sidebar.slider("Cooling duty (% of stack power)", 2, 35, 12)
st.sidebar.markdown("### Delivery & economics")
inlet_bar = st.sidebar.number_input("Compressor inlet pressure (bar)", 1.0, 100.0, 30.0)
outlet_bar = st.sidebar.number_input("Target H₂ pressure (bar)", 10.0, 1000.0, 350.0)
compressor_eff_pct = st.sidebar.slider("Compressor isothermal efficiency (%)", 40, 90, 70)
electricity_price = st.sidebar.number_input("Electricity price (currency/kWh)", 0.0, 100.0, 6.0, 0.25)
water_price = st.sidebar.number_input("DI water cost (currency/m³)", 0.0, 10000.0, 150.0, 10.0)
capex_per_kw = st.sidebar.number_input("Electrolyzer installed CAPEX (currency/kW)", 1000, 200000, 65000, 1000)
capex_per_kg_day = st.sidebar.number_input("Water system CAPEX (currency per m³/day)", 1000, 10000000, 100000, 1000)
capex_other = st.sidebar.number_input("Other CAPEX (currency)", 0, 10000000000, 5000000, 500000)
fixed_opex_pct = st.sidebar.slider("Fixed OPEX (% total CAPEX/year)", 1, 15, 4) / 100
project_years = st.sidebar.slider("Project life (years)", 5, 30, 20)
discount_rate = st.sidebar.slider("Discount rate (%)", 0, 20, 8) / 100

d = calculate_design(
    capacity, specific_energy, uptime_pct / 100, water_factor,
    cooling_fraction_pct / 100, outlet_bar, inlet_bar, compressor_eff_pct / 100,
    electricity_price, water_price, capex_per_kw, capex_per_kg_day,
    capex_other, fixed_opex_pct, 80000, project_years, discount_rate
)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div><span class="pill">AI-ASSISTED ENGINEERING</span><span class="pill">CONCEPTUAL DESIGN</span></div>
  <h1 style="margin:12px 0 6px 0;">GreenH₂ Studio</h1>
  <p class="muted" style="font-size:16px;margin:0;">
  From design basis to process flow, utilities and first-pass techno-economics.
  </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Overview", "Process Flow Diagram", "Utilities & Sizing",
    "Techno-economics", "Digital Twin", "AI Engineering Copilot", "Data & Aspen Export"
])

# ---------- Overview ----------
with tabs[0]:
    st.subheader("Plant overview")
    st.caption(f"{electrolyzer} electrolyzer · {capacity:,.0f} kg/day nameplate · conceptual design basis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("H₂ production", f"{d['h2_daily']:,.0f} kg/day", f"{d['h2_hourly']:.1f} kg/h")
    c2.metric("Electrolyzer load", f"{d['power_kw']:,.0f} kW")
    c3.metric("DI water demand", f"{d['water_m3_day']:.2f} m³/day")
    c4.metric("Indicative LCOH", f"{d['lcoh']:,.0f} /kg")
    st.markdown("#### Process at a glance")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.graphviz_chart(pfd_to_dot(default_pfd()), use_container_width=True)
    with col2:
        st.markdown("**Key design assumptions**")
        st.write(f"- Specific electricity: **{specific_energy:.1f} kWh/kg H₂**")
        st.write(f"- Availability: **{uptime_pct}%**")
        st.write(f"- Water factor: **{water_factor:.1f}×** stoichiometric")
        st.write(f"- Delivery pressure: **{outlet_bar:.0f} bar**")
        st.write(f"- Project life: **{project_years} years**")
        st.info("Use these results for early screening only. Replace assumptions with vendor guarantees and validated process calculations before engineering decisions.")
    st.markdown("#### Estimated daily energy split")
    energy_df = pd.DataFrame({
        "Load": ["Electrolyzer", "H₂ compression"],
        "kWh/day": [d["electricity_kwh_day"], d["compressor_kwh_day"]],
    })
    fig = px.bar(energy_df, x="Load", y="kWh/day", color="Load", title=None)
    fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------- PFD ----------
with tabs[1]:
    st.subheader("AI-generated process flow diagram")
    st.write("Generate a process-flow specification from your design basis, then render it as a diagram.")
    if "pfd_spec" not in st.session_state:
        st.session_state.pfd_spec = default_pfd()
    if st.button("✨ Generate PFD with Gemini", type="primary", key="gen_pfd"):
        prompt = f"""
You are a process engineer creating a PRELIMINARY green hydrogen PFD specification.
Return ONLY valid JSON with keys title, nodes, edges.
nodes: array of 6-12 objects with id and label.
edges: array of objects with from, to, label. Every edge endpoint must match a node id.
Include DI water treatment, electrolyzer, gas/liquid separation, hydrogen purification/drying,
compression and storage; show oxygen by-product and cooling/power inputs where appropriate.
Do not claim the design is validated. Keep IDs alphanumeric/underscore.
Plant basis: technology={electrolyzer}, nameplate={capacity} kg H2/day,
specific electricity={specific_energy} kWh/kg, delivery pressure={outlet_bar} bar.
Labels should be short. JSON only.
"""
        with st.spinner("Asking Gemini to draft the PFD..."):
            try:
                spec = extract_json(gemini_text(prompt))
                if not isinstance(spec.get("nodes"), list) or not isinstance(spec.get("edges"), list):
                    raise ValueError("JSON must contain nodes and edges arrays.")
                st.session_state.pfd_spec = spec
                st.success("PFD specification generated. Review before use.")
            except Exception as e:
                st.error(f"Could not generate PFD: {e}")
    try:
        st.graphviz_chart(pfd_to_dot(st.session_state.pfd_spec), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render generated PFD: {e}")
    st.download_button(
        "Download PFD specification (JSON)",
        data=json.dumps(st.session_state.pfd_spec, indent=2),
        file_name="green_hydrogen_pfd.json",
        mime="application/json",
    )
    st.caption("This is a conceptual PFD, not a P&ID. It does not contain validated stream properties, control loops, relief design or safety analysis.")

# ---------- Utilities and equipment sizing ----------
with tabs[2]:
    st.subheader("Utilities and first-pass equipment sizing")
    st.caption("Calculated locally from the design basis; Gemini is not used for the numeric calculations.")
    a, b, c = st.columns(3)
    a.metric("Electrolyzer nameplate", f"{d['power_kw']:,.0f} kW")
    b.metric("Cooling duty estimate", f"{d['cooling_kw']:,.0f} kWₜₕ")
    c.metric("Compressor power estimate", f"{d['compressor_kw']:,.1f} kW")
    utility_df = pd.DataFrame([
        ["Hydrogen production", f"{d['h2_daily']:,.1f}", "kg/day"],
        ["Electrolyzer electricity", f"{d['electricity_kwh_day']:,.1f}", "kWh/day"],
        ["Compression electricity", f"{d['compressor_kwh_day']:,.1f}", "kWh/day"],
        ["DI water demand", f"{d['water_m3_day']:,.2f}", "m³/day"],
        ["Cooling duty", f"{d['cooling_kw']:,.1f}", "kW thermal"],
        ["Compressor power", f"{d['compressor_kw']:,.2f}", "kW"],
        ["Stoichiometric water consumption", f"{d['h2_daily'] * 9:,.1f}", "kg/day"],
        ["Oxygen co-product (theoretical)", f"{d['h2_daily'] * 8:,.1f}", "kg/day"],
    ], columns=["Parameter", "Estimate", "Unit"])
    st.dataframe(utility_df, use_container_width=True, hide_index=True)
    st.markdown("#### Equipment sizing notes")
    st.write(f"- **Electrolyzer:** approximately {d['power_kw']:,.0f} kW nameplate input at the selected specific electricity consumption.")
    st.write(f"- **Water treatment:** minimum design throughput is approximately {d['water_m3_day']:.2f} m³/day; add storage, regeneration and peak-flow margin during detailed design.")
    st.write(f"- **Hydrogen compressor:** screening estimate uses an ideal-gas isothermal-work approximation from {inlet_bar:g} to {outlet_bar:g} bar. Real multi-stage compressors need vendor curves, cooling, pressure drops and gas-property calculations.")
    st.write(f"- **Cooling:** preliminary duty is {cooling_fraction_pct}% of electrolyzer electrical load; convert to cooling-water flow only after selecting supply/return temperatures and heat-exchanger approach.")
    with st.expander("Calculation assumptions and limitations"):
        st.markdown("""
        - Hydrogen stoichiometry: 9 kg water consumed and 8 kg oxygen co-produced per kg H₂.
        - Electrolyzer electricity is a user-entered system-specific assumption.
        - DI-water design factor accounts approximately for process losses; it is not a water-treatment guarantee.
        - Cooling duty is a user-set fraction of stack input, not a heat balance.
        - Compression estimate assumes hydrogen as an ideal gas at 298.15 K and an isothermal efficiency. It is not a compressor vendor sizing calculation.
        - No stack degradation, part-load efficiency, pressure drop, purge loss, water recycle or detailed thermodynamics are modelled.
        """)

# ---------- TEA ----------
with tabs[3]:
    st.subheader("Techno-economic analysis")
    st.caption("Illustrative screening model. Currency follows your entered prices; no currency conversion is applied.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated total CAPEX", f"{d['total_capex']:,.0f}")
    m2.metric("Annual operating cost", f"{d['annual_opex']:,.0f}")
    m3.metric("Levelized H₂ cost", f"{d['lcoh']:,.0f} /kg")
    cost_df = pd.DataFrame([
        ["Electricity", d["annual_electricity_cost"]],
        ["DI water", d["annual_water_cost"]],
        ["Fixed OPEX allowance", d["annual_fixed_opex"]],
        ["Annualized CAPEX", d["annualized_capex"]],
    ], columns=["Cost component", "Annual cost"])
    fig = px.pie(cost_df, names="Cost component", values="Annual cost", hole=0.55)
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### CAPEX breakdown")
    capex_df = pd.DataFrame([
        ["Electrolyzer", d["electrolyzer_capex"]],
        ["Compression allowance", d["compression_capex"]],
        ["Water system allowance", d["water_capex"]],
        ["Other CAPEX", capex_other],
    ], columns=["Component", "CAPEX"])
    fig2 = px.bar(capex_df, x="Component", y="CAPEX", color="Component")
    fig2.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(cost_df, use_container_width=True, hide_index=True)
    st.warning("CAPEX inputs are placeholder screening assumptions, not market quotes. LCOH excludes taxes, financing structure, replacement schedules, oxygen revenue, transport, storage losses and many project-specific costs.")

# ---------- Digital twin ----------
with tabs[4]:
    st.subheader("Operational digital-twin concept")
    st.write("Adjust the operating scenario to see how estimated production and utilities respond. This is a simulated dashboard, not a live plant connection.")
    twin_load = st.slider("Stack load (%)", 10, 110, 90, key="twin_load")
    twin_availability = st.slider("Current plant availability (%)", 0, 100, 95, key="twin_availability")
    twin_eff_delta = st.slider("Specific energy deviation (%)", -15, 30, 0, key="twin_eff")
    load_fraction = twin_load / 100
    twin_h2 = capacity * (twin_availability / 100) * load_fraction
    twin_energy = specific_energy * (1 + twin_eff_delta / 100)
    twin_power = capacity * load_fraction * twin_energy / 24
    twin_water = twin_h2 * 9 * water_factor / 1000
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Estimated H₂", f"{twin_h2:,.1f} kg/day")
    x2.metric("Stack power", f"{twin_power:,.1f} kW")
    x3.metric("DI water", f"{twin_water:,.2f} m³/day")
    x4.metric("Specific energy", f"{twin_energy:.1f} kWh/kg")
    hours = list(range(24))
    profile = [max(0, math.sin(math.pi * (h + 1) / 24)) * twin_power for h in hours]
    profile_df = pd.DataFrame({"Hour": hours, "Illustrative stack power (kW)": profile})
    fig = px.area(profile_df, x="Hour", y="Illustrative stack power (kW)")
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The displayed 24-hour curve is illustrative only. Real digital-twin capability requires time-series sensor data, calibrated models, alarms and validation.")

# ---------- Copilot ----------
with tabs[5]:
    st.subheader("AI engineering copilot")
    st.write("Ask for conceptual recommendations, assumptions to verify, or help interpreting the screening results.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input("Ask about electrolyzer selection, utilities, compression, or project economics...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        context = f"""
You are GreenH2 Studio's engineering copilot. Give practical, clearly qualified preliminary guidance.
Do not fabricate standards, vendor data, Aspen results, citations, or safety approvals. State assumptions
and request missing inputs when needed. Never present preliminary estimates as validated engineering.
Recommend professional review for detailed design, hazardous-area classification, pressure systems,
relief sizing, process safety, or construction.
Current design basis:
technology={electrolyzer}; nameplate={capacity} kg H2/day; specific energy={specific_energy} kWh/kg;
availability={uptime_pct}%; estimated production={d['h2_daily']:.2f} kg/day;
electrolyzer power={d['power_kw']:.2f} kW; DI water={d['water_m3_day']:.3f} m3/day;
cooling duty={d['cooling_kw']:.2f} kW thermal; compression={inlet_bar} to {outlet_bar} bar;
compressor power estimate={d['compressor_kw']:.2f} kW; illustrative LCOH={d['lcoh']:.2f} per kg.
User question: {question}
"""
        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking..."):
                try:
                    answer = gemini_text(context)
                except Exception as e:
                    answer = f"Gemini request failed: {e}"
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ---------- Data & Aspen export ----------
with tabs[6]:
    st.subheader("Performance data repository")
    st.write("This MVP includes editable assumptions and calculated KPIs. Add vendor datasheets and validated site data to turn this into a real engineering repository.")
    default_data = pd.DataFrame([
        ["Alkaline", "Illustrative", 52.0, 30.0, "Replace with vendor datasheet"],
        ["PEM", "Illustrative", 55.0, 30.0, "Replace with vendor datasheet"],
        ["SOEC", "Illustrative", 42.0, 10.0, "Highly system-dependent; validate"],
    ], columns=["Technology", "Data status", "Example kWh/kg H₂", "Example outlet bar", "Notes"])
    edited_data = st.data_editor(default_data, use_container_width=True, num_rows="dynamic", key="performance_repo")
    st.download_button(
        "Download performance repository CSV",
        edited_data.to_csv(index=False).encode("utf-8"),
        "electrolyzer_performance_repository.csv",
        "text/csv",
    )

    st.markdown("#### Export design basis for simulation handoff")
    st.write("Export a JSON handoff package containing inputs, assumptions, and first-pass calculations. It is not an Aspen Plus `.apwz` file and does not run an Aspen simulation.")
    handoff = {
        "project": "GreenH2 Studio conceptual design",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PRELIMINARY_NOT_VALIDATED",
        "software_handoff_note": "Import/build this design manually or via a separately developed Aspen automation workflow. This JSON is not a native Aspen file.",
        "design_basis": {
            "technology": electrolyzer, "nameplate_h2_kg_day": capacity,
            "specific_electricity_kwh_per_kg": specific_energy,
            "availability_fraction": uptime_pct / 100, "water_design_factor": water_factor,
            "inlet_pressure_bar": inlet_bar, "outlet_pressure_bar": outlet_bar,
            "compressor_efficiency_fraction": compressor_eff_pct / 100,
        },
        "calculated_screening_results": d,
        "pfd_specification": st.session_state.get("pfd_spec", default_pfd()),
        "limitations": [
            "No rigorous thermodynamic property package or stream flash calculations.",
            "No vendor equipment models or validated electrolyzer polarization curve.",
            "No pressure-drop, relief, controls, HAZOP or process-safety design.",
            "Not an Aspen Plus .bkp/.apwz or Aspen HYSYS native project.",
        ],
    }
    st.download_button(
        "Download simulation handoff JSON",
        json.dumps(handoff, indent=2),
        "green_hydrogen_simulation_handoff.json",
        "application/json",
        type="primary",
    )
    st.download_button(
        "Download KPI table CSV",
        pd.DataFrame([{"Metric": k, "Value": v} for k, v in d.items()]).to_csv(index=False),
        "green_hydrogen_kpis.csv",
        "text/csv",
    )

st.markdown("---")
st.markdown(
    '<p class="muted" style="font-size:12px;">GreenH₂ Studio MVP · AI-assisted conceptual design only. '
    'All calculations and economic estimates require independent engineering review before use in a real plant.</p>',
    unsafe_allow_html=True,
)
