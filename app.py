"""
GETT · NYC Fleet Command Center  —  "Sunset Pastel" edition
Interactive taxi-demand forecasting dashboard.

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
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="side-h">When in the city?</div>', unsafe_allow_html=True)
    hour = st.slider("Hour of day", 0, 23, 18, format="%d:00")
    st.markdown(f'<div class="side-cap">Showing <b>{fmt_hour(hour)}</b></div>', unsafe_allow_html=True)
    st.write("")
    day_name = st.selectbox("Day of week", DAYS, index=4)
    day = DAYS.index(day_name)
    st.write("")
    big_map = st.toggle("Enlarge the map", value=False)
    st.markdown('<div class="side-foot">Forecasts from a tuned LightGBM model · '
                'trained on 2024–2025 NYC TLC trips · test MAE 3.17</div>', unsafe_allow_html=True)


def mood(h):
    if h <= 5:  return "linear-gradient(165deg,#ECE7F6 0%,#F4EAF1 100%)"
    if h <= 9:  return "linear-gradient(165deg,#FFEDE3 0%,#FCE6EF 100%)"
    if h <= 16: return "linear-gradient(165deg,#FFF6EE 0%,#FDEFE7 100%)"
    if h <= 20: return "linear-gradient(165deg,#FFE5D9 0%,#F3DFEF 100%)"
    return "linear-gradient(165deg,#EDE5F1 0%,#E8E7F4 100%)"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
:root {{ --peach:#FF8B6B; --peach-d:#F2724E; --pink:#F2A0BE; --sky:#79B0D6;
  --ink:#3D3450; --muted:#9387A2; --surface:#FFFFFF; --line:#F2E5DC; }}
.stApp {{ background:{mood(hour)}; transition:background 1.2s ease; }}
.block-container {{ padding-top:1.3rem; padding-bottom:2.5rem; max-width:1380px; }}
html, body, [class*="css"] {{ font-family:'Plus Jakarta Sans',sans-serif; color:var(--ink); }}
@keyframes rise {{ from{{opacity:0;transform:translateY(8px);}} to{{opacity:1;transform:translateY(0);}} }}
.rise {{ animation:rise .5s cubic-bezier(.2,.7,.3,1) both; }}
.hero {{ position:relative; overflow:hidden;
  background:linear-gradient(120deg,#FF8B6B 0%,#F2799C 52%,#9F7BC9 100%);
  border-radius:26px; padding:26px 32px 56px; margin-bottom:22px;
  box-shadow:0 20px 44px -20px rgba(242,114,120,.55); }}
.hero h1 {{ font-family:'Fraunces',serif; font-weight:600; color:#fff; font-size:1.9rem;
  margin:0; letter-spacing:-.4px; line-height:1.05; }}
.hero p {{ color:rgba(255,255,255,.9); margin:5px 0 0; font-size:.92rem; font-weight:500; }}
.hero-row {{ display:flex; align-items:center; gap:16px; position:relative; z-index:2; }}
.hero-mark {{ width:54px; height:54px; border-radius:16px; background:rgba(255,255,255,.18);
  display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.pill {{ margin-left:auto; display:flex; align-items:center; gap:8px; background:rgba(255,255,255,.2);
  color:#fff; padding:9px 16px; border-radius:100px; font-size:.78rem; font-weight:600; }}
.dot {{ width:8px; height:8px; border-radius:50%; background:#C6FFE6;
  box-shadow:0 0 0 0 rgba(198,255,230,.7); animation:pulse 1.8s infinite; }}
@keyframes pulse {{ 0%{{box-shadow:0 0 0 0 rgba(198,255,230,.6);}}
  70%{{box-shadow:0 0 0 10px rgba(198,255,230,0);}} 100%{{box-shadow:0 0 0 0 rgba(198,255,230,0);}} }}
.skyline {{ position:absolute; left:0; right:0; bottom:0; width:100%; height:46px; z-index:1; opacity:.4; }}
.mcard {{ background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:18px 20px;
  height:100%; box-shadow:0 8px 22px -16px rgba(157,123,201,.5); }}
.mtop {{ display:flex; align-items:center; gap:9px; color:var(--muted); font-size:.74rem;
  font-weight:600; text-transform:uppercase; letter-spacing:.5px; }}
.mval {{ font-family:'Fraunces',serif; font-weight:600; font-size:2rem; color:var(--ink);
  margin-top:8px; line-height:1.05; }}
.msub {{ font-size:.8rem; color:var(--muted); margin-top:2px; font-weight:500; }}
.ic {{ width:18px; height:18px; fill:none; stroke-width:2; }}
.sect {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.22rem; color:var(--ink);
  margin:8px 0 12px; display:flex; align-items:center; gap:10px; }}
.insight {{ background:linear-gradient(105deg,#FFF1EA 0%,#FDEAF1 100%); border:1px solid #FBD9CE;
  border-left:5px solid var(--peach); border-radius:18px; padding:17px 20px; font-size:.98rem;
  color:#5e4435; line-height:1.55; font-weight:500; }}
.insight b {{ color:var(--peach-d); }}
.zcard {{ background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:14px 16px;
  box-shadow:0 8px 22px -17px rgba(157,123,201,.5); }}
.zrank {{ font-family:'Fraunces',serif; font-weight:600; font-size:1rem; color:var(--peach-d); }}
.zname {{ font-weight:700; font-size:.95rem; margin:2px 0 1px; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }}
.zboro {{ font-size:.76rem; color:var(--muted); font-weight:500; }}
.zval {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.45rem; margin-top:8px; }}
.zbar {{ height:6px; border-radius:100px; background:#F4E9E2; margin-top:8px; overflow:hidden; }}
.zfill {{ height:100%; border-radius:100px; background:linear-gradient(90deg,var(--sky),var(--peach)); }}
.tip {{ background:linear-gradient(105deg,#EAF3FA 0%,#F3ECFA 100%); border:1px solid #D9E7F4;
  border-radius:18px; padding:16px 19px; font-size:.95rem; color:#3f4a63; font-weight:500;
  line-height:1.5; display:flex; gap:12px; align-items:flex-start; }}
section[data-testid="stSidebar"] {{ background:#FFFDFB; border-right:1px solid var(--line); }}
.side-h {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.05rem; color:var(--ink); margin-bottom:6px; }}
.side-cap {{ font-size:.82rem; color:var(--muted); margin-top:-4px; }}
.side-foot {{ font-size:.74rem; color:var(--muted); margin-top:18px; line-height:1.5;
  border-top:1px solid var(--line); padding-top:12px; }}
footer, #MainMenu {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
cur = pred[(pred["hour"] == hour) & (pred["day_of_week"] == day)].copy()
cur["zone"] = cur["PULocationID"].map(ZONE_NAME)
cur["borough"] = cur["PULocationID"].map(ZONE_BORO)
total_demand = int(cur["predicted_demand"].sum())
busy_zones = int((cur["predicted_demand"] >= 50).sum())
top = cur.sort_values("predicted_demand", ascending=False).reset_index(drop=True)
hottest = top.iloc[0]
day_curve = pred[pred["day_of_week"] == day].groupby("hour")["predicted_demand"].sum()
peak_hour = int(day_curve.idxmax())

TAXI = ('<svg width="30" height="30" viewBox="0 0 24 24" fill="#fff">'
        '<path d="M3.5 13l1.3-4A2.3 2.3 0 017 7.4h10a2.3 2.3 0 012.2 1.6l1.3 4"/>'
        '<rect x="2.6" y="12.6" width="18.8" height="5.4" rx="1.6"/>'
        '<rect x="9" y="5.6" width="6" height="2.1" rx=".5"/>'
        '<circle cx="7.2" cy="18.4" r="1.7" fill="#3D3450"/>'
        '<circle cx="16.8" cy="18.4" r="1.7" fill="#3D3450"/></svg>')

SKYLINE = ('<svg class="skyline" viewBox="0 0 1200 80" preserveAspectRatio="none" fill="#fff">'
           '<rect x="40" y="40" width="40" height="40"/><rect x="90" y="22" width="34" height="58"/>'
           '<rect x="132" y="48" width="44" height="32"/><rect x="250" y="30" width="30" height="50"/>'
           '<rect x="286" y="14" width="20" height="66"/><rect x="312" y="44" width="40" height="36"/>'
           '<rect x="470" y="36" width="36" height="44"/><rect x="512" y="8" width="24" height="72"/>'
           '<rect x="542" y="40" width="34" height="40"/><rect x="690" y="46" width="40" height="34"/>'
           '<rect x="736" y="26" width="28" height="54"/><rect x="900" y="38" width="34" height="42"/>'
           '<rect x="940" y="18" width="22" height="62"/><rect x="968" y="48" width="46" height="32"/>'
           '<rect x="1090" y="34" width="36" height="46"/></svg>')

IC = {
 "car":'<svg class="ic" viewBox="0 0 24 24" stroke="#FF8B6B"><path d="M5 13l1.4-4.2A2 2 0 018.3 7.5h7.4a2 2 0 011.9 1.3L19 13"/><rect x="4" y="13" width="16" height="5" rx="1.4"/><circle cx="7.5" cy="18" r="1.3"/><circle cx="16.5" cy="18" r="1.3"/></svg>',
 "flame":'<svg class="ic" viewBox="0 0 24 24" stroke="#F2724E"><path d="M12 3c1 3 4 4 4 8a4 4 0 01-8 0c0-1.6.6-2.6 1.1-3.1.2 1 .8 1.6 1.6 1.6C11 8 10 6 12 3z"/></svg>',
 "grid":'<svg class="ic" viewBox="0 0 24 24" stroke="#F2A0BE"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></svg>',
 "clock":'<svg class="ic" viewBox="0 0 24 24" stroke="#79B0D6"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
 "map":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#F2724E" stroke-width="2"><path d="M9 5l6 2 6-2v12l-6 2-6-2-6 2V7z"/><path d="M9 5v12M15 7v12"/></svg>',
 "pin":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#F2A0BE" stroke-width="2"><path d="M12 21s-6-5.3-6-10a6 6 0 1112 0c0 4.7-6 10-6 10z"/><circle cx="12" cy="11" r="2"/></svg>',
 "pulse":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#79B0D6" stroke-width="2"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>',
 "bulb":'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#79B0D6" stroke-width="2"><path d="M9 18h6M10 21h4M12 3a6 6 0 00-4 10.5c.7.7 1 1.3 1 2.5h6c0-1.2.3-1.8 1-2.5A6 6 0 0012 3z"/></svg>',
}

st.markdown(f"""
<div class="hero rise"><div class="hero-row">
  <div class="hero-mark">{TAXI}</div>
  <div><h1>NYC Fleet Command Center</h1>
  <p>Gett · predictive taxi-demand across New York City</p></div>
  <div class="pill"><span class="dot"></span>FORECAST LIVE</div>
</div>{SKYLINE}</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
cards = [
 (m1, IC["car"], "Total demand", f"{total_demand:,}", f"{fmt_hour(hour)} · {day_name}"),
 (m2, IC["flame"], "Hottest zone", ZONE_NAME.get(int(hottest['PULocationID']), '—'),
    f"{ZONE_BORO.get(int(hottest['PULocationID']),'')} · {hottest['predicted_demand']:.0f} rides"),
 (m3, IC["grid"], "Busy zones", f"{busy_zones}", "zones above 50 rides/hr"),
 (m4, IC["clock"], "Citywide peak", fmt_hour(peak_hour), f"busiest hour · {day_name}"),
]
for col, icon, label, val, sub in cards:
    col.markdown(f'<div class="mcard rise"><div class="mtop">{icon}{label}</div>'
                 f'<div class="mval">{val}</div><div class="msub">{sub}</div></div>',
                 unsafe_allow_html=True)

st.write("")
left, right = st.columns([2.15, 1])

with left:
    st.markdown(f'<div class="sect">{IC["map"]} Demand map · {fmt_hour(hour)}, {day_name}</div>',
                unsafe_allow_html=True)
    scale = [[0.0,"#E9F3FA"],[0.22,"#FFE7CC"],[0.46,"#FFC9A3"],
             [0.68,"#FF9E84"],[0.85,"#F58AA9"],[1.0,"#E0507F"]]
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson, locations=cur["PULocationID"], z=cur["predicted_demand"],
        featureidkey="id", colorscale=scale, marker_opacity=0.85,
        marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.65)",
        colorbar=dict(title="rides/hr", thickness=13, len=0.82,
                      tickfont=dict(family="Plus Jakarta Sans", size=11, color="#9387A2")),
        customdata=np.stack([cur["zone"], cur["borough"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>Demand: %{z:.0f} rides/hr<extra></extra>"))
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=9.3,
        mapbox_center={"lat":40.730,"lon":-73.945}, margin=dict(l=0,r=0,t=0,b=0),
        height=720 if big_map else 530, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True,
        config={"displayModeBar":True,"displaylogo":False,
                "modeBarButtonsToRemove":["lasso2d","select2d"]})

    share = hottest["predicted_demand"]/max(total_demand,1)*100
    fleet_pct = min(45, max(8, round(share*4)))
    st.markdown(
        f'<div class="insight rise">At <b>{fmt_hour(hour)} on {day_name}</b>, demand peaks in '
        f'<b>{ZONE_NAME.get(int(hottest["PULocationID"]),"")}</b> '
        f'({ZONE_BORO.get(int(hottest["PULocationID"]),"")}) — about '
        f'<b>{hottest["predicted_demand"]:.0f} rides/hr</b>. Position roughly '
        f'<b>{fleet_pct}%</b> of available drivers across the hottest zones to meet it.</div>',
        unsafe_allow_html=True)

with right:
    st.markdown(f'<div class="sect">{IC["pin"]} Top 5 hottest zones</div>', unsafe_allow_html=True)
    top5 = top.head(5)
    maxv = top5["predicted_demand"].max()
    for i, row in top5.iterrows():
        lid = int(row["PULocationID"])
        fill = row["predicted_demand"]/maxv*100
        st.markdown(
            f'<div class="zcard rise" style="margin-bottom:10px;"><div class="zrank">#{i+1}</div>'
            f'<div class="zname">{ZONE_NAME.get(lid,"Zone")}</div>'
            f'<div class="zboro">{ZONE_BORO.get(lid,"")}</div>'
            f'<div class="zval">{row["predicted_demand"]:.0f}'
            f'<span style="font-size:.7rem;color:var(--muted);font-family:Plus Jakarta Sans;"> rides/hr</span></div>'
            f'<div class="zbar"><div class="zfill" style="width:{fill}%"></div></div></div>',
            unsafe_allow_html=True)


def driver_tip(h, d):
    weekend = d >= 5
    if h <= 5: return "Quiet pre-dawn hours. Keep a lean fleet near airports and hospitals."
    if 6 <= h <= 9 and not weekend: return "Morning commute is ramping up — shift drivers toward Manhattan business districts."
    if 17 <= h <= 19: return "Evening rush is the daily peak. Concentrate drivers in Midtown and the Upper East Side."
    if h >= 21 and weekend: return "Weekend nightlife is alive — keep drivers near downtown and entertainment zones."
    if weekend and 10 <= h <= 16: return "Relaxed weekend daytime. Spread drivers across leisure and shopping areas."
    return "Steady demand. Balance the fleet between Manhattan cores and the outer boroughs."

st.write("")
st.markdown(f'<div class="tip rise">{IC["bulb"]}<div><b>Driver tip · {fmt_hour(hour)}, {day_name}</b>'
            f'<br>{driver_tip(hour, day)}</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown(f'<div class="sect">{IC["pulse"]} Pulse of the day · {day_name}</div>', unsafe_allow_html=True)
pulse = go.Figure()
pulse.add_trace(go.Scatter(x=list(range(24)), y=day_curve.values, mode="lines",
    line=dict(color="#FF8B6B", width=3.5, shape="spline"),
    fill="tozeroy", fillcolor="rgba(255,139,107,0.12)",
    hovertemplate="%{x}:00 — %{y:,.0f} rides<extra></extra>"))
pulse.add_trace(go.Scatter(x=[hour], y=[day_curve.values[hour]], mode="markers",
    marker=dict(color="#E0507F", size=16, line=dict(color="#fff", width=3)),
    hovertemplate=f"{fmt_hour(hour)} — %{{y:,.0f}} rides<extra></extra>"))
pulse.update_layout(height=235, margin=dict(l=10,r=10,t=10,b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    xaxis=dict(tickmode="array", tickvals=list(range(0,24,3)),
               ticktext=[fmt_hour(h) for h in range(0,24,3)], showgrid=False,
               tickfont=dict(family="Plus Jakarta Sans", size=11, color="#9387A2")),
    yaxis=dict(showgrid=True, gridcolor="#F2E5DC", zeroline=False,
               tickfont=dict(family="Plus Jakarta Sans", size=11, color="#9387A2")))
st.plotly_chart(pulse, use_container_width=True, config={"displayModeBar": False})
