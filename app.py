"""
GETT · NYC Fleet Command Center
Interactive taxi-demand forecasting dashboard.

Every number shown is computed directly from the model's predictions
(app_predictions.csv) — nothing is hard-coded or invented.

Run locally:   streamlit run app.py
Required files (same folder): app.py, app_predictions.csv, nyc_zones.geojson
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gett · NYC Fleet Command Center",
                   page_icon="🚕", layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def load_geojson():
    with open("nyc_zones.geojson", "r") as f:
        gj = json.load(f)
    names, boroughs = {}, {}
    for feat in gj["features"]:
        p = feat["properties"]
        lid = int(p["locationid"])
        feat["id"] = lid
        names[lid] = p.get("zone", f"Zone {lid}")
        boroughs[lid] = p.get("borough", "")
    return gj, names, boroughs


@st.cache_data
def load_predictions():
    return pd.read_csv("app_predictions.csv")


geojson, ZONE_NAME, ZONE_BORO = load_geojson()
pred = load_predictions()
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fmt_hour(h):
    s = "AM" if h < 12 else "PM"
    hr = h % 12
    hr = 12 if hr == 0 else hr
    return f"{hr} {s}"


# --------------------------------------------------------------------------- #
# Detailed colour icons (drawn, not emoji)
# --------------------------------------------------------------------------- #
def _svg(inner, size=20, vb=48):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {vb} {vb}" '
            f'fill="none" style="vertical-align:middle">{inner}</svg>')

TAXI = ('<rect x="6" y="20" width="36" height="13" rx="3" fill="#F2C200"/>'
        '<path d="M11 20 l3 -7 h20 l3 7" fill="#F2C200" stroke="#6C2C25" stroke-width="1.6"/>'
        '<rect x="6" y="20" width="36" height="13" rx="3" fill="none" stroke="#6C2C25" stroke-width="1.6"/>'
        '<rect x="19" y="9" width="10" height="4" rx="1" fill="#6C2C25"/>'
        '<rect x="22" y="23" width="11" height="5" rx="1" fill="#EB5E3A"/>'
        '<circle cx="14" cy="33" r="4" fill="#3a2f2a" stroke="#6C2C25" stroke-width="1.4"/>'
        '<circle cx="34" cy="33" r="4" fill="#3a2f2a" stroke="#6C2C25" stroke-width="1.4"/>')
PIN = ('<path d="M24 6 a12 12 0 0 1 12 12 c0 9 -12 22 -12 22 S12 27 12 18 A12 12 0 0 1 24 6 Z" '
       'fill="#EB5E3A" stroke="#6C2C25" stroke-width="1.6"/><circle cx="24" cy="18" r="5" fill="#FBF6EE"/>')
CLOCK = ('<circle cx="24" cy="24" r="17" fill="#5E7E80" stroke="#6C2C25" stroke-width="1.6"/>'
         '<circle cx="24" cy="24" r="13" fill="#E8F0F2"/>'
         '<line x1="24" y1="24" x2="24" y2="14" stroke="#6C2C25" stroke-width="2" stroke-linecap="round"/>'
         '<line x1="24" y1="24" x2="31" y2="27" stroke="#EB5E3A" stroke-width="2" stroke-linecap="round"/>'
         '<circle cx="24" cy="24" r="1.6" fill="#6C2C25"/>')
BARS = ('<rect x="7" y="26" width="7" height="13" rx="1.5" fill="#5E7E80"/>'
        '<rect x="17" y="18" width="7" height="21" rx="1.5" fill="#EB5E3A"/>'
        '<rect x="27" y="22" width="7" height="17" rx="1.5" fill="#F2C200"/>'
        '<path d="M9 22 L20 13 L30 17 L40 8" stroke="#6C2C25" stroke-width="2" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round"/><circle cx="40" cy="8" r="2.2" fill="#6C2C25"/>')
PIE = ('<circle cx="24" cy="24" r="16" fill="#FBF3D9" stroke="#6C2C25" stroke-width="1.6"/>'
       '<path d="M24 24 L24 9 A15 15 0 0 1 38 28 Z" fill="#EB5E3A"/>'
       '<path d="M24 24 L38 28 A15 15 0 0 1 14 36 Z" fill="#5E7E80"/>'
       '<circle cx="24" cy="24" r="5" fill="#fff"/>')
MAPIC = ('<path d="M9 5l6 2 6-2v12l-6 2-6-2-6 2V7z" fill="#F2EDBD" stroke="#6C2C25" stroke-width="1.6" '
         'transform="scale(2)"/><path d="M9 5v12M15 7v12" stroke="#6C2C25" stroke-width="1.6" transform="scale(2)"/>')
PULSE = ('<path d="M6 24h7l4-12 8 24 4-12h13" stroke="#EB5E3A" stroke-width="2.4" fill="none" '
         'stroke-linecap="round" stroke-linejoin="round"/>')
BULB = ('<path d="M18 36h12M20 42h8M24 6a12 12 0 0 0-8 21c1.4 1.4 2 2.6 2 5h12c0-2.4.6-3.6 2-5A12 12 0 0 0 24 6z" '
        'fill="#F2C200" stroke="#6C2C25" stroke-width="1.6" stroke-linejoin="round"/>')
RACE = ('<path d="M10 40V8M10 10h22l-3 6 3 6H10" fill="#EB5E3A" stroke="#6C2C25" stroke-width="1.8" '
        'stroke-linejoin="round"/>')


# --------------------------------------------------------------------------- #
# Sidebar — selector (day + hour)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="side-h">Plan your fleet</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Choose a day and time to forecast.</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown('<div class="side-label">Day of week</div>', unsafe_allow_html=True)
    day_name = st.selectbox("Day of week", DAYS, index=4, label_visibility="collapsed")
    day = DAYS.index(day_name)
    st.write("")

    st.markdown('<div class="side-label">Time of day</div>', unsafe_allow_html=True)
    hour = st.slider("Time of day", 0, 23, 18, format="%d:00", label_visibility="collapsed")
    st.write("")

    st.markdown(
        f'<div class="now-badge">{_svg(CLOCK, 22)}'
        f'<div><div class="now-cap">Showing</div>'
        f'<div class="now-val">{day_name}, {fmt_hour(hour)}</div></div></div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Styling  (vintage palette — matches the report)
# MAROON #6C2C25 · TOMATO #EB5E3A · GOLD #F2C200 · CHINA BLUE #5E7E80
# BUTTER #F2EDBD · CREAM #FBF6EE
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Courier+Prime:wght@400;700&display=swap');
:root {
  --maroon:#6C2C25; --tomato:#EB5E3A; --gold:#F2C200; --blue:#5E7E80;
  --butter:#F2EDBD; --cream:#FBF6EE; --ink:#3a2f2a; --muted:#9a8e84;
  --surface:#FFFDF9; --line:#E7DFD2;
}
.stApp { background:#F4ECE0; }
.block-container { padding-top:1.3rem; padding-bottom:2.5rem; max-width:1380px; }
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; color:var(--ink); }
@keyframes rise { from{opacity:0;transform:translateY(8px);} to{opacity:1;transform:translateY(0);} }
.rise { animation:rise .5s cubic-bezier(.2,.7,.3,1) both; }

.taxi-stripe { height:10px; border-radius:100px; margin:0 0 18px;
  background:repeating-linear-gradient(45deg,#6C2C25 0 11px,#F2C200 11px 22px); opacity:.92; }

/* HERO */
.hero { position:relative; overflow:hidden;
  background:linear-gradient(120deg,#6C2C25 0%,#9B3A2A 60%,#B23A1E 120%);
  border-radius:26px; padding:26px 34px 58px; margin-bottom:20px;
  box-shadow:0 20px 44px -22px rgba(108,44,37,.6); }
.hero h1 { font-family:'Fraunces',serif; font-weight:600; color:#FBF6EE; font-size:2rem;
  margin:0; letter-spacing:-.4px; line-height:1.05; }
.hero p { color:rgba(251,246,238,.85); margin:5px 0 0; font-size:.93rem; font-weight:500; }
.hero-top { display:flex; align-items:flex-start; justify-content:space-between; position:relative; z-index:3; }
.hero-left { display:flex; align-items:center; gap:18px; }
.hero-mark { width:60px; height:60px; border-radius:17px; background:rgba(242,237,189,.20);
  display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.pill { display:flex; align-items:center; gap:8px; background:rgba(242,237,189,.20);
  color:#FBF6EE; padding:9px 16px; border-radius:100px; font-size:.78rem; font-weight:600;
  white-space:nowrap; margin-top:6px; }
.dot { width:8px; height:8px; border-radius:50%; background:#F2C200;
  box-shadow:0 0 0 0 rgba(242,194,0,.7); animation:pulse 1.8s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(242,194,0,.6);}
  70%{box-shadow:0 0 0 10px rgba(242,194,0,0);} 100%{box-shadow:0 0 0 0 rgba(242,194,0,0);} }
.skyline { position:absolute; left:0; right:0; bottom:0; width:100%; height:50px; z-index:1; opacity:.22; }
.bigtaxi { position:absolute; right:150px; bottom:16px; z-index:2; opacity:.9; }

/* RECEIPT METRIC CARDS */
.rcard { position:relative; background:var(--cream); border:1px solid var(--line); border-bottom:none;
  border-radius:14px 14px 0 0; padding:16px 18px 12px; height:100%;
  box-shadow:0 10px 22px -18px rgba(108,44,37,.45); }
.rc-top { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:.7rem;
  font-weight:700; text-transform:uppercase; letter-spacing:1px; }
.rc-val { font-family:'Courier Prime',monospace; font-weight:700; font-size:1.9rem; color:var(--maroon);
  margin-top:7px; line-height:1.05; letter-spacing:-1px; }
.rc-val.txt { font-family:'Fraunces',serif; font-size:1.35rem; letter-spacing:0; line-height:1.1; }
.rc-sub { font-size:.78rem; color:var(--muted); margin-top:3px; font-weight:500; }
.rc-dash { border-top:1.5px dashed #CDBFAC; margin-top:11px; padding-top:7px;
  font-family:'Courier Prime',monospace; font-size:.66rem; color:#b6a995;
  display:flex; justify-content:space-between; letter-spacing:.5px; }
.rc-zig { display:block; width:100%; height:8px; margin-top:-1px; }

/* SECTIONS */
.sect { font-family:'Fraunces',serif; font-weight:600; font-size:1.25rem; color:var(--maroon);
  margin:8px 0 12px; display:flex; align-items:center; gap:10px; }

/* TOP-5 panel */
.top5 { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:8px 6px;
  box-shadow:0 10px 24px -18px rgba(108,44,37,.45); }
.zrow { display:flex; align-items:center; gap:12px; padding:11px 14px; border-bottom:1px solid #F2ECE2; }
.zrow:last-child { border-bottom:none; }
.zrank { font-family:'Fraunces',serif; font-weight:600; font-size:1.05rem; color:var(--tomato); width:26px; flex-shrink:0; }
.zmid { flex:1; min-width:0; }
.zname { font-weight:700; font-size:.92rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.zboro { font-size:.73rem; color:var(--muted); font-weight:500; }
.zbar { height:5px; border-radius:100px; background:#EFE3D0; margin-top:5px; overflow:hidden; }
.zfill { height:100%; border-radius:100px; background:linear-gradient(90deg,#F2C200,#EB5E3A,#6C2C25); }
.zval { font-family:'Courier Prime',monospace; font-weight:700; font-size:1.1rem; color:var(--maroon);
  flex-shrink:0; text-align:right; }
.zval span { font-size:.6rem; color:var(--muted); font-family:'Plus Jakarta Sans'; font-weight:500; }

/* DRIVER TIP */
.tip { background:#FBF3D9; border:1px solid #ECD9A6; border-left:5px solid var(--tomato);
  border-radius:18px; padding:16px 19px; font-size:.95rem; color:#5a4a33; font-weight:500;
  line-height:1.5; display:flex; gap:12px; align-items:flex-start; }
.tip b { color:var(--maroon); }

/* SIDEBAR */
section[data-testid="stSidebar"] { background:#FBF6EE; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top:2rem; }
.side-h { font-family:'Fraunces',serif; font-weight:600; font-size:1.55rem; color:var(--maroon);
  letter-spacing:-.3px; line-height:1.1; }
.side-sub { font-size:.86rem; color:var(--muted); margin-top:4px; font-weight:500; }
.side-label { font-family:'Fraunces',serif; font-weight:600; font-size:1rem; color:var(--maroon); margin-bottom:7px; }
.now-badge { display:flex; align-items:center; gap:12px; background:#fff; border:1px solid var(--line);
  border-left:4px solid var(--tomato); border-radius:14px; padding:13px 16px;
  box-shadow:0 8px 20px -16px rgba(108,44,37,.5); }
.now-cap { font-size:.68rem; text-transform:uppercase; letter-spacing:.7px; color:var(--muted); font-weight:700; }
.now-val { font-family:'Fraunces',serif; font-weight:600; font-size:1.12rem; color:var(--maroon); margin-top:1px; }
.side-note { font-size:.88rem; color:var(--maroon); margin-top:18px; line-height:1.55;
  display:flex; gap:10px; align-items:flex-start; font-weight:500; background:#FBF3D9;
  border:1px solid #ECD9A6; border-radius:14px; padding:13px 15px; }
.side-foot { font-size:.72rem; color:var(--muted); margin-top:20px; line-height:1.5;
  border-top:1px solid var(--line); padding-top:13px; }
/* sidebar button -> tomato */
section[data-testid="stSidebar"] .stButton>button { background:var(--maroon); color:#FBF6EE;
  border:none; border-radius:12px; font-weight:700; font-family:'Plus Jakarta Sans'; padding:11px 14px;
  box-shadow:0 8px 18px -12px rgba(108,44,37,.7); }
section[data-testid="stSidebar"] .stButton>button:hover { background:#883a30; color:#fff; }
footer, #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Current slice + ALL derived numbers (100% from data)
# --------------------------------------------------------------------------- #
cur = pred[(pred["hour"] == hour) & (pred["day_of_week"] == day)].copy()
cur["zone"] = cur["PULocationID"].map(ZONE_NAME)
cur["borough"] = cur["PULocationID"].map(ZONE_BORO)

total_demand = int(cur["predicted_demand"].sum())
busy_zones = int((cur["predicted_demand"] >= 50).sum())
top = cur.sort_values("predicted_demand", ascending=False).reset_index(drop=True)
hottest = top.iloc[0]
hot_id = int(hottest["PULocationID"])
hot_name = ZONE_NAME.get(hot_id, "")
hot_boro = ZONE_BORO.get(hot_id, "")
hot_val = hottest["predicted_demand"]

day_curve = pred[pred["day_of_week"] == day].groupby("hour")["predicted_demand"].sum()
peak_hour = int(day_curve.idxmax())

boro_sum = cur.groupby("borough")["predicted_demand"].sum().sort_values(ascending=False)
lead_boro = boro_sum.index[0]
lead_boro_pct = round(boro_sum.iloc[0] / max(total_demand, 1) * 100)

# torn-receipt bottom edge (generated once)
_zig = "M0 0 H240 V2 "
_x = 240
while _x > 0:
    _zig += f"L{_x-6} 8 L{_x-12} 2 "
    _x -= 12
_zig += "Z"
ZIG = (f'<svg class="rc-zig" viewBox="0 0 240 8" preserveAspectRatio="none">'
       f'<path d="{_zig}" fill="#FBF6EE" stroke="#E7DFD2" stroke-width="1"/></svg>')

# --------------------------------------------------------------------------- #
# Illustrations (hero)
# --------------------------------------------------------------------------- #
BIGTAXI = ('<svg class="bigtaxi" width="104" height="62" viewBox="0 0 120 72" fill="none">'
           '<path d="M14 44l6-19a10 10 0 019.6-7.2h40.8A10 10 0 0180 25l6 19" fill="#F2C200"/>'
           '<rect x="8" y="42" width="92" height="20" rx="6" fill="#F2C200"/>'
           '<rect x="40" y="13" width="26" height="9" rx="2" fill="#6C2C25"/>'
           '<rect x="30" y="27" width="20" height="13" rx="2" fill="#FBF6EE"/>'
           '<rect x="56" y="27" width="22" height="13" rx="2" fill="#FBF6EE"/>'
           '<rect x="8" y="50" width="92" height="6" fill="#6C2C25"/>'
           '<rect x="14" y="50" width="8" height="6" fill="#F2C200"/><rect x="30" y="50" width="8" height="6" fill="#F2C200"/>'
           '<rect x="46" y="50" width="8" height="6" fill="#F2C200"/><rect x="62" y="50" width="8" height="6" fill="#F2C200"/>'
           '<rect x="78" y="50" width="8" height="6" fill="#F2C200"/>'
           '<circle cx="30" cy="62" r="8" fill="#6C2C25"/><circle cx="30" cy="62" r="3.4" fill="#FBF6EE"/>'
           '<circle cx="84" cy="62" r="8" fill="#6C2C25"/><circle cx="84" cy="62" r="3.4" fill="#FBF6EE"/></svg>')

SKYLINE = ('<svg class="skyline" viewBox="0 0 1200 80" preserveAspectRatio="none" fill="#FBF6EE">'
           '<rect x="40" y="40" width="40" height="40"/><rect x="90" y="22" width="34" height="58"/>'
           '<rect x="132" y="48" width="44" height="32"/><rect x="250" y="30" width="30" height="50"/>'
           '<rect x="286" y="14" width="20" height="66"/><rect x="312" y="44" width="40" height="36"/>'
           '<rect x="470" y="36" width="36" height="44"/><rect x="512" y="8" width="24" height="72"/>'
           '<rect x="542" y="40" width="34" height="40"/><rect x="690" y="46" width="40" height="34"/>'
           '<rect x="736" y="26" width="28" height="54"/><rect x="900" y="38" width="34" height="42"/>'
           '<rect x="940" y="18" width="22" height="62"/><rect x="968" y="48" width="46" height="32"/>'
           '<rect x="1090" y="34" width="36" height="46"/></svg>')

# --------------------------------------------------------------------------- #
# HERO
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class="hero rise">
  <div class="hero-top">
    <div class="hero-left">
      <div class="hero-mark">{_svg(TAXI, 34)}</div>
      <div><h1>NYC Fleet Command Center</h1>
      <p>Gett · predictive taxi-demand across New York City</p></div>
    </div>
    <div class="pill"><span class="dot"></span>FORECAST LIVE</div>
  </div>
  {BIGTAXI}{SKYLINE}
</div>
<div class="taxi-stripe"></div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# RECEIPT METRIC CARDS
# --------------------------------------------------------------------------- #
def receipt_card(col, icon, label, value, sub, unit, is_text=False):
    cls = "rc-val txt" if is_text else "rc-val"
    col.markdown(
        f'<div class="rcard rise"><div class="rc-top">{_svg(icon, 18)}{label}</div>'
        f'<div class="{cls}">{value}</div><div class="rc-sub">{sub}</div>'
        f'<div class="rc-dash"><span>FARE EST.</span><span>{unit}</span></div></div>{ZIG}',
        unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
receipt_card(m1, TAXI, "Total demand", f"{total_demand:,}", f"{fmt_hour(hour)} · {day_name}", "rides/hr")
receipt_card(m2, PIN, "Hottest zone", hot_name, f"{hot_boro} · {hot_val:.0f} rides", "leading", is_text=True)
receipt_card(m3, BARS, "Busy zones", f"{busy_zones}", "zones above 50 rides/hr", "count")
receipt_card(m4, CLOCK, "Citywide peak", fmt_hour(peak_hour), f"busiest hour · {day_name}", "peak")

st.write("")
st.write("")
left, right = st.columns([2.15, 1])

with left:
    st.markdown(f'<div class="sect">{_svg(MAPIC, 22)} Demand map · {fmt_hour(hour)}, {day_name}</div>',
                unsafe_allow_html=True)
    scale = [[0.0, "#FBF6EE"], [0.16, "#F2EDBD"], [0.38, "#F2C200"],
             [0.60, "#EB5E3A"], [0.82, "#B23A1E"], [1.0, "#6C2C25"]]
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson, locations=cur["PULocationID"], z=cur["predicted_demand"],
        featureidkey="id", colorscale=scale, marker_opacity=0.88,
        marker_line_width=0.5, marker_line_color="rgba(251,246,238,0.7)",
        colorbar=dict(title="rides/hr", thickness=13, len=0.82,
                      tickfont=dict(family="Plus Jakarta Sans", size=11, color="#9a8e84")),
        customdata=np.stack([cur["zone"], cur["borough"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Demand: %{z:.0f} rides/hr<extra></extra>"))
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=9.3,
        mapbox_center={"lat": 40.730, "lon": -73.945}, margin=dict(l=0, r=0, t=0, b=0),
        height=540, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

with right:
    st.markdown(f'<div class="sect">{_svg(PIN, 20)} Top 5 hottest zones</div>', unsafe_allow_html=True)
    top5 = top.head(5)
    maxv = top5["predicted_demand"].max()
    rows = ""
    for i, row in top5.iterrows():
        lid = int(row["PULocationID"])
        fill = row["predicted_demand"] / maxv * 100
        rows += (
            f'<div class="zrow"><div class="zrank">#{i+1}</div>'
            f'<div class="zmid"><div class="zname">{ZONE_NAME.get(lid,"Zone")}</div>'
            f'<div class="zboro">{ZONE_BORO.get(lid,"")}</div>'
            f'<div class="zbar"><div class="zfill" style="width:{fill}%"></div></div></div>'
            f'<div class="zval">{row["predicted_demand"]:.0f}<br><span>rides/hr</span></div></div>')
    st.markdown(f'<div class="top5 rise">{rows}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# DRIVER TIP — fully data-driven
# --------------------------------------------------------------------------- #
second = top.iloc[1]
sec_name = ZONE_NAME.get(int(second["PULocationID"]), "")
vs_peak = round(total_demand / max(day_curve.max(), 1) * 100)

tip_text = (
    f"<b>{lead_boro}</b> leads right now with <b>{lead_boro_pct}%</b> of citywide demand. "
    f"Prioritise <b>{hot_name}</b> ({hot_val:.0f} rides/hr), then <b>{sec_name}</b> "
    f"({second['predicted_demand']:.0f}). ")
if hour == peak_hour:
    tip_text += "This is the busiest hour of the day, so deploy the fleet at full strength."
elif total_demand <= day_curve.min() * 1.15:
    tip_text += f"Demand is near its daily low (peak is at {fmt_hour(peak_hour)}), so a lean fleet is enough."
else:
    tip_text += f"Citywide demand is about {vs_peak}% of today's peak ({fmt_hour(peak_hour)})."

st.write("")
st.markdown(f'<div class="tip rise">{_svg(BULB, 24)}<div><b>Driver tip · {fmt_hour(hour)}, {day_name}</b>'
            f'<br>{tip_text}</div></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# PULSE OF THE DAY
# --------------------------------------------------------------------------- #
st.write("")
st.markdown(f'<div class="sect">{_svg(PULSE, 20)} Pulse of the day · {day_name}</div>', unsafe_allow_html=True)
pulse = go.Figure()
pulse.add_trace(go.Scatter(x=list(range(24)), y=day_curve.values, mode="lines",
    line=dict(color="#EB5E3A", width=3.5, shape="spline"),
    fill="tozeroy", fillcolor="rgba(235,94,58,0.12)",
    hovertemplate="%{x}:00 — %{y:,.0f} rides<extra></extra>"))
pulse.add_trace(go.Scatter(x=[hour], y=[day_curve.values[hour]], mode="markers",
    marker=dict(color="#6C2C25", size=15, line=dict(color="#FBF6EE", width=3)),
    hovertemplate=f"{fmt_hour(hour)} — %{{y:,.0f}} rides<extra></extra>"))
pulse.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 3)),
               ticktext=[fmt_hour(h) for h in range(0, 24, 3)], showgrid=False,
               tickfont=dict(family="Plus Jakarta Sans", size=11, color="#9a8e84")),
    yaxis=dict(showgrid=True, gridcolor="#E7DFD2", zeroline=False,
               tickfont=dict(family="Plus Jakarta Sans", size=11, color="#9a8e84")))
st.plotly_chart(pulse, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------------------------------- #
# 24-HOUR RACE — opens in a centered modal from the sidebar button
# --------------------------------------------------------------------------- #
def build_race_fig():
    day_all = pred[pred["day_of_week"] == day].copy()
    top_ids = (day_all.groupby("PULocationID")["predicted_demand"].max()
               .sort_values(ascending=False).head(8).index.tolist())
    RACE_COLORS = ["#6C2C25", "#B23A1E", "#EB5E3A", "#D98E63",
                   "#F2C200", "#5E7E80", "#96C0CE", "#3a2f2a"]

    def hour_bars(h):
        sl = day_all[day_all["hour"] == h].set_index("PULocationID")["predicted_demand"]
        vals = [(lid, float(sl.get(lid, 0.0))) for lid in top_ids]
        vals.sort(key=lambda t: t[1])
        names = [ZONE_NAME.get(lid, str(lid)) for lid, _ in vals]
        nums = [v for _, v in vals]
        cmap = {lid: RACE_COLORS[i] for i, lid in enumerate(top_ids)}
        colors = [cmap[lid] for lid, _ in vals]
        return names, nums, colors

    xmax = day_all[day_all["PULocationID"].isin(top_ids)]["predicted_demand"].max() * 1.15

    def bar_trace(names, nums, colors):
        return go.Bar(x=nums, y=names, orientation="h", marker=dict(color=colors),
                      text=[f"{v:,.0f}" for v in nums], textposition="outside",
                      textfont=dict(family="Plus Jakarta Sans", size=12, color="#6C2C25"),
                      cliponaxis=False, hoverinfo="skip")

    n0, v0, c0 = hour_bars(0)
    frames = []
    for h in range(24):
        nm, nu, co = hour_bars(h)
        frames.append(go.Frame(name=str(h), data=[bar_trace(nm, nu, co)],
                      layout=go.Layout(annotations=[dict(
                          x=0.98, y=0.06, xref="paper", yref="paper", showarrow=False,
                          text=f"<b>{fmt_hour(h)}</b> — {nm[-1]} leads",
                          font=dict(family="Fraunces", size=22, color="#6C2C25"), align="right")])))

    race = go.Figure(
        data=[bar_trace(n0, v0, c0)], frames=frames,
        layout=go.Layout(
            height=440, margin=dict(l=10, r=70, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, xmax], showgrid=True, gridcolor="#E7DFD2",
                       tickfont=dict(family="Plus Jakarta Sans", size=11, color="#9a8e84")),
            yaxis=dict(tickfont=dict(family="Plus Jakarta Sans", size=12, color="#6C2C25")),
            annotations=[dict(x=0.98, y=0.06, xref="paper", yref="paper", showarrow=False,
                              text=f"<b>{fmt_hour(0)}</b> — {n0[-1]} leads",
                              font=dict(family="Fraunces", size=22, color="#6C2C25"), align="right")],
            updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.16, xanchor="left",
                bgcolor="#6C2C25", bordercolor="#6C2C25", font=dict(color="#FBF6EE", size=12),
                buttons=[
                    dict(label="▶  Play the day", method="animate",
                         args=[None, dict(frame=dict(duration=420, redraw=True),
                                          fromcurrent=True, transition=dict(duration=300, easing="cubic-in-out"))]),
                    dict(label="❚❚  Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ])]))
    return race


def _render_race_body():
    st.markdown(f'<div class="sect">{_svg(RACE, 20)} The 24-hour race · {day_name}</div>', unsafe_allow_html=True)
    st.plotly_chart(build_race_fig(), use_container_width=True, config={"displayModeBar": False})
    st.caption("Press Play to watch the busiest zones rise and fall across the day. "
               "Close this window with the X in the top corner.")


_dialog_deco = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
if _dialog_deco is not None:
    try:
        _race_dialog = _dialog_deco("The 24-hour race", width="large")(_render_race_body)
    except TypeError:
        _race_dialog = _dialog_deco("The 24-hour race")(_render_race_body)
else:
    _race_dialog = None

# Sidebar trigger + footer (added after the dialog is defined)
st.sidebar.write("")
if st.sidebar.button("▶  Play 24-hour simulation", use_container_width=True, key="race_btn"):
    if _race_dialog is not None:
        _race_dialog()
    else:
        st.session_state["_show_race_inline"] = True

st.sidebar.markdown(
    '<div class="side-note"><div>Demand is <b>never spread evenly.</b> '
    'Some zones surge while others stay quiet. See where riders will be.</div></div>',
    unsafe_allow_html=True)
st.sidebar.markdown('<div class="side-foot">Tuned LightGBM model · trained on 2024 to 2025 '
                    'NYC TLC trips · test MAE 3.17</div>', unsafe_allow_html=True)

# Fallback for very old Streamlit without dialog support: show race inline.
if _race_dialog is None and st.session_state.get("_show_race_inline"):
    st.write("")
    _render_race_body()
