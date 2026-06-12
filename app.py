"""
GETT · NYC Fleet Command Center
Interactive taxi-demand forecasting dashboard.

Run locally:   streamlit run app.py
Required files (same folder):
    app.py
    app_predictions.csv     (generated in the notebook)
    nyc_zones.geojson       (NYC taxi-zone polygons)
"""

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- #
# Page config
# ----------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Gett · NYC Fleet Command Center",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------- #
# Design tokens + global styling
# ----------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --bg:      #F4F6F8;
        --surface: #FFFFFF;
        --ink:     #15233B;
        --muted:   #6B7687;
        --teal:    #0E8C8B;
        --teal-d:  #0A6D6C;
        --coral:   #FF6B4A;
        --line:    #E6E9EE;
    }

    .stApp { background: var(--bg); }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

    /* ---- Header ---- */
    .hero {
        display: flex; align-items: center; justify-content: space-between;
        background: linear-gradient(110deg, #0E8C8B 0%, #0A6D6C 60%, #114E54 100%);
        border-radius: 20px; padding: 22px 30px; margin-bottom: 22px;
        box-shadow: 0 12px 30px -12px rgba(14,140,139,.45);
    }
    .hero-left { display: flex; align-items: center; gap: 16px; }
    .hero-mark {
        width: 46px; height: 46px; border-radius: 13px; background: rgba(255,255,255,.16);
        display: flex; align-items: center; justify-content: center;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #fff;
        font-size: 1.45rem; margin: 0; letter-spacing: -.3px; line-height: 1.1;
    }
    .hero p { color: rgba(255,255,255,.82); margin: 2px 0 0; font-size: .85rem; }
    .live-pill {
        display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,.14);
        color: #fff; padding: 8px 15px; border-radius: 100px; font-size: .78rem;
        font-weight: 600; letter-spacing: .3px;
    }
    .live-dot { width: 8px; height: 8px; border-radius: 50%; background: #6EF7C0;
                box-shadow: 0 0 0 0 rgba(110,247,192,.7); animation: pulse 1.8s infinite; }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(110,247,192,.6); }
        70%  { box-shadow: 0 0 0 9px rgba(110,247,192,0); }
        100% { box-shadow: 0 0 0 0 rgba(110,247,192,0); }
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
        padding: 16px 18px; height: 100%;
        box-shadow: 0 4px 14px -10px rgba(21,35,59,.25);
    }
    .metric-top { display: flex; align-items: center; gap: 9px; color: var(--muted);
                  font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; }
    .metric-val { font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.85rem;
                  color: var(--ink); margin-top: 6px; line-height: 1.1; }
    .metric-sub { font-size: .78rem; color: var(--muted); margin-top: 1px; }
    .ico { width: 17px; height: 17px; stroke: var(--teal); fill: none; stroke-width: 2; }
    .ico-coral { stroke: var(--coral); }

    /* ---- Section labels ---- */
    .sect { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem;
            color: var(--ink); margin: 6px 0 10px; display: flex; align-items: center; gap: 9px; }

    /* ---- Insight banner ---- */
    .insight {
        background: linear-gradient(100deg, #FFF4F1 0%, #FFFBF5 100%);
        border: 1px solid #FFD9CE; border-left: 4px solid var(--coral);
        border-radius: 14px; padding: 15px 18px; font-size: .94rem; color: #5a2a1c;
        margin-top: 4px; line-height: 1.5;
    }
    .insight b { color: var(--coral); }

    /* ---- Zone cards ---- */
    .zone-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
        padding: 13px 15px; box-shadow: 0 4px 14px -11px rgba(21,35,59,.25);
    }
    .zone-rank { font-family: 'Space Mono', monospace; font-size: .72rem; color: var(--teal);
                 font-weight: 700; }
    .zone-name { font-weight: 600; font-size: .92rem; color: var(--ink); margin: 3px 0 1px;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .zone-boro { font-size: .73rem; color: var(--muted); }
    .zone-val  { font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.2rem;
                 color: var(--ink); margin-top: 7px; }
    .zone-bar  { height: 5px; border-radius: 100px; background: var(--line); margin-top: 7px; overflow: hidden; }
    .zone-fill { height: 100%; border-radius: 100px;
                 background: linear-gradient(90deg, var(--teal), var(--coral)); }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
    .side-title { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:.8rem;
                  text-transform:uppercase; letter-spacing:.7px; color:var(--muted); margin: 4px 0 2px; }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- #
# Inline SVG icons (lucide-style)
# ----------------------------------------------------------------------------- #
IC = {
    "clock": '<svg class="ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    "car": '<svg class="ico" viewBox="0 0 24 24"><path d="M5 13l1.5-4.5A2 2 0 018.4 7h7.2a2 2 0 011.9 1.5L19 13"/><path d="M5 13h14v4a1 1 0 01-1 1h-1a1 1 0 01-1-1v-1H8v1a1 1 0 01-1 1H6a1 1 0 01-1-1z"/><circle cx="7.5" cy="15.5" r="1"/><circle cx="16.5" cy="15.5" r="1"/></svg>',
    "flame": '<svg class="ico ico-coral" viewBox="0 0 24 24"><path d="M12 3c1 3 4 4 4 8a4 4 0 01-8 0c0-1.5.5-2.5 1-3 .2 1 .8 1.5 1.5 1.5C11 8 10 6 12 3z"/></svg>',
    "grid": '<svg class="ico" viewBox="0 0 24 24"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></svg>',
    "map": '<svg style="width:18px;height:18px;stroke:#15233B;fill:none;stroke-width:2" viewBox="0 0 24 24"><path d="M9 5l6 2 6-2v12l-6 2-6-2-6 2V7z"/><path d="M9 5v12M15 7v12"/></svg>',
    "pulse": '<svg style="width:18px;height:18px;stroke:#0E8C8B;fill:none;stroke-width:2" viewBox="0 0 24 24"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>',
    "pin": '<svg style="width:18px;height:18px;stroke:#FF6B4A;fill:none;stroke-width:2" viewBox="0 0 24 24"><path d="M12 21s-6-5.3-6-10a6 6 0 1112 0c0 4.7-6 10-6 10z"/><circle cx="12" cy="11" r="2"/></svg>',
}


# ----------------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------------- #
@st.cache_data
def load_geojson():
    with open("nyc_zones.geojson", "r") as f:
        gj = json.load(f)
    names, boroughs = {}, {}
    for feat in gj["features"]:
        p = feat["properties"]
        lid = int(p["locationid"])
        feat["id"] = lid  # set top-level id for plotly matching
        names[lid] = p.get("zone", f"Zone {lid}")
        boroughs[lid] = p.get("borough", "")
    return gj, names, boroughs


@st.cache_data
def load_predictions():
    df = pd.read_csv("app_predictions.csv")
    return df


geojson, ZONE_NAME, ZONE_BORO = load_geojson()
pred = load_predictions()

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fmt_hour(h):
    suffix = "AM" if h < 12 else "PM"
    hr = h % 12
    hr = 12 if hr == 0 else hr
    return f"{hr} {suffix}"


# ----------------------------------------------------------------------------- #
# Sidebar controls
# ----------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="side-title">Controls</div>', unsafe_allow_html=True)

    # Advance hour automatically if "play" is active (runs before slider is built)
    if st.session_state.get("playing"):
        st.session_state.sel_hour = (st.session_state.get("sel_hour", 18) + 1) % 24

    st.markdown("**Time of day**")
    hour = st.slider("Hour", 0, 23, key="sel_hour", value=18,
                     format="%d:00", label_visibility="collapsed")
    st.caption(f"Showing demand at **{fmt_hour(hour)}**")

    st.markdown("**Day of week**")
    day_name = st.selectbox("Day", DAYS, index=4, label_visibility="collapsed")
    day = DAYS.index(day_name)

    st.divider()
    play = st.toggle("▶  Play the day", key="playing",
                     help="Animate demand across all 24 hours")

    st.divider()
    st.caption("Forecasts from a tuned LightGBM model · "
               "trained on 2024–2025 NYC TLC trips · test MAE 3.17")


# ----------------------------------------------------------------------------- #
# Filter + enrich data for the current selection
# ----------------------------------------------------------------------------- #
cur = pred[(pred["hour"] == hour) & (pred["day_of_week"] == day)].copy()
cur["zone"] = cur["PULocationID"].map(ZONE_NAME)
cur["borough"] = cur["PULocationID"].map(ZONE_BORO)

total_demand = int(cur["predicted_demand"].sum())
busy_zones = int((cur["predicted_demand"] >= 50).sum())
top = cur.sort_values("predicted_demand", ascending=False).reset_index(drop=True)
hottest = top.iloc[0]

# Citywide peak hour for this day (for a metric)
day_curve = (pred[pred["day_of_week"] == day]
             .groupby("hour")["predicted_demand"].sum())
peak_hour = int(day_curve.idxmax())


# ----------------------------------------------------------------------------- #
# Header
# ----------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-left">
        <div class="hero-mark">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2">
            <path d="M5 13l1.5-4.5A2 2 0 018.4 7h7.2a2 2 0 011.9 1.5L19 13"/>
            <path d="M5 13h14v4a1 1 0 01-1 1h-1a1 1 0 01-1-1v-1H8v1a1 1 0 01-1 1H6a1 1 0 01-1-1z"/>
            <circle cx="7.5" cy="15.5" r="1.2"/><circle cx="16.5" cy="15.5" r="1.2"/>
          </svg>
        </div>
        <div>
          <h1>NYC Fleet Command Center</h1>
          <p>Gett · predictive taxi-demand intelligence across New York City</p>
        </div>
      </div>
      <div class="live-pill"><span class="live-dot"></span>FORECAST LIVE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- #
# Metric cards
# ----------------------------------------------------------------------------- #
m1, m2, m3, m4 = st.columns(4)
cards = [
    (m1, IC["car"], "Total demand", f"{total_demand:,}", f"{fmt_hour(hour)} · {day_name}"),
    (m2, IC["flame"], "Hottest zone", ZONE_NAME.get(int(hottest['PULocationID']), '—'),
        f"{ZONE_BORO.get(int(hottest['PULocationID']), '')} · {hottest['predicted_demand']:.0f} rides"),
    (m3, IC["grid"], "Busy zones", f"{busy_zones}", "zones above 50 rides/hr"),
    (m4, IC["clock"], "Citywide peak", fmt_hour(peak_hour), f"busiest hour on {day_name}"),
]
for col, icon, label, val, sub in cards:
    col.markdown(
        f"""<div class="metric-card">
              <div class="metric-top">{icon}{label}</div>
              <div class="metric-val">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------------- #
# Map + side column
# ----------------------------------------------------------------------------- #
left, right = st.columns([2.1, 1])

with left:
    st.markdown(f'<div class="sect">{IC["map"]} Demand map · {fmt_hour(hour)}, {day_name}</div>',
                unsafe_allow_html=True)

    colorscale = [
        [0.00, "#E8F4F3"], [0.20, "#9FE0CF"], [0.40, "#F2D06B"],
        [0.62, "#FF9E4A"], [0.82, "#FF6B4A"], [1.00, "#D62828"],
    ]

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=cur["PULocationID"],
        z=cur["predicted_demand"],
        featureidkey="id",
        colorscale=colorscale,
        marker_opacity=0.82,
        marker_line_width=0.4,
        marker_line_color="rgba(255,255,255,0.5)",
        colorbar=dict(title="Rides/hr", thickness=12, len=0.8,
                      tickfont=dict(family="Space Mono", size=10)),
        customdata=np.stack([cur["zone"], cur["borough"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>Demand: %{z:.0f} rides/hr<extra></extra>",
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9.3,
        mapbox_center={"lat": 40.730, "lon": -73.945},
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Auto insight ----
    share = hottest["predicted_demand"] / max(total_demand, 1) * 100
    fleet_pct = min(45, max(8, round(share * 4)))
    st.markdown(
        f"""<div class="insight">
        At <b>{fmt_hour(hour)} on {day_name}</b>, demand concentrates in
        <b>{ZONE_NAME.get(int(hottest['PULocationID']),'')}</b>
        ({ZONE_BORO.get(int(hottest['PULocationID']),'')}), with about
        <b>{hottest['predicted_demand']:.0f} rides/hr</b>. Consider positioning
        roughly <b>{fleet_pct}%</b> of available drivers across the top zones below
        to meet demand efficiently.
        </div>""",
        unsafe_allow_html=True,
    )

with right:
    st.markdown(f'<div class="sect">{IC["pin"]} Top 5 hottest zones</div>',
                unsafe_allow_html=True)
    top5 = top.head(5)
    maxv = top5["predicted_demand"].max()
    for i, row in top5.iterrows():
        lid = int(row["PULocationID"])
        fill = row["predicted_demand"] / maxv * 100
        st.markdown(
            f"""<div class="zone-card" style="margin-bottom:9px;">
                  <div class="zone-rank">#{i+1}</div>
                  <div class="zone-name">{ZONE_NAME.get(lid,'Zone')}</div>
                  <div class="zone-boro">{ZONE_BORO.get(lid,'')}</div>
                  <div class="zone-val">{row['predicted_demand']:.0f}
                     <span style="font-size:.7rem;color:var(--muted);">rides/hr</span></div>
                  <div class="zone-bar"><div class="zone-fill" style="width:{fill}%"></div></div>
                </div>""",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------- #
# Pulse of the day (signature element)
# ----------------------------------------------------------------------------- #
st.write("")
st.markdown(f'<div class="sect">{IC["pulse"]} Pulse of the day · {day_name}</div>',
            unsafe_allow_html=True)

pulse = go.Figure()
pulse.add_trace(go.Scatter(
    x=list(range(24)), y=day_curve.values,
    mode="lines", line=dict(color="#0E8C8B", width=3, shape="spline"),
    fill="tozeroy", fillcolor="rgba(14,140,139,0.10)",
    hovertemplate="%{x}:00 — %{y:,.0f} rides<extra></extra>",
))
pulse.add_trace(go.Scatter(
    x=[hour], y=[day_curve.values[hour]],
    mode="markers", marker=dict(color="#FF6B4A", size=15,
                                line=dict(color="#fff", width=2)),
    hovertemplate=f"{fmt_hour(hour)} — %{{y:,.0f}} rides<extra></extra>",
))
pulse.update_layout(
    height=230, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 3)),
               ticktext=[fmt_hour(h) for h in range(0, 24, 3)],
               showgrid=False, tickfont=dict(family="Space Mono", size=10, color="#6B7687")),
    yaxis=dict(showgrid=True, gridcolor="#EEF1F4", zeroline=False,
               tickfont=dict(family="Space Mono", size=10, color="#6B7687")),
)
st.plotly_chart(pulse, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------- #
# Animation loop for "Play the day"
# ----------------------------------------------------------------------------- #
if play:
    time.sleep(0.55)
    st.rerun()
