"""
GETT · NYC Fleet Command Center  —  "Coastal" edition
Interactive taxi-demand forecasting dashboard.

Run locally:   streamlit run app.py
Required files (same folder): app.py, app_predictions.csv, nyc_zones.geojson
"""

import json
import datetime as dt
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
# Palette (Coastal)
# TAN #D8BA98 · MAROON #7F0303 · ALABASTER #EFE8DF · LIGHT BLUE #96C0CE · MIDNIGHT #0F414A
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="side-h">Plan your fleet</div>', unsafe_allow_html=True)
    picked_date = st.date_input("Date", value=dt.date(2026, 1, 16))
    day = picked_date.weekday()           # 0=Mon
    day_name = DAYS[day]
    st.markdown(f'<div class="side-cap">{day_name}</div>', unsafe_allow_html=True)
    st.write("")
    hour = st.slider("Time of day", 0, 23, 18, format="%d:00")
    st.markdown(f'<div class="side-cap">Showing <b>{fmt_hour(hour)}</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-foot">Forecasts from a tuned LightGBM model · '
                'trained on 2024–2025 NYC TLC trips · test MAE 3.17</div>', unsafe_allow_html=True)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
:root {
  --tan:#D8BA98; --maroon:#7F0303; --alabaster:#EFE8DF;
  --blue:#96C0CE; --midnight:#0F414A; --ink:#0F414A; --muted:#7C8A8E;
  --surface:#FFFDF9; --line:#E7DFD2;
}
.stApp { background:#EFE8DF; }
.block-container { padding-top:1.3rem; padding-bottom:2.5rem; max-width:1380px; }
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; color:var(--ink); }
@keyframes rise { from{opacity:0;transform:translateY(8px);} to{opacity:1;transform:translateY(0);} }
.rise { animation:rise .5s cubic-bezier(.2,.7,.3,1) both; }

/* checkerboard taxi stripe */
.taxi-stripe { height:10px; border-radius:100px; margin:0 0 18px;
  background:repeating-linear-gradient(45deg,#0F414A 0 11px,#F2C200 11px 22px); opacity:.9; }

/* HERO */
.hero { position:relative; overflow:hidden;
  background:linear-gradient(120deg,#0F414A 0%,#155A63 55%,#7F0303 130%);
  border-radius:26px; padding:28px 34px 58px; margin-bottom:20px;
  box-shadow:0 20px 44px -22px rgba(15,65,74,.6); }
.hero h1 { font-family:'Fraunces',serif; font-weight:600; color:#EFE8DF; font-size:2rem;
  margin:0; letter-spacing:-.4px; line-height:1.05; }
.hero p { color:rgba(239,232,223,.85); margin:5px 0 0; font-size:.93rem; font-weight:500; }
.hero-row { display:flex; align-items:center; gap:18px; position:relative; z-index:3; }
.hero-mark { width:60px; height:60px; border-radius:17px; background:rgba(216,186,152,.22);
  display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.pill { margin-left:auto; display:flex; align-items:center; gap:8px; background:rgba(216,186,152,.22);
  color:#EFE8DF; padding:9px 16px; border-radius:100px; font-size:.78rem; font-weight:600; }
.dot { width:8px; height:8px; border-radius:50%; background:#96C0CE;
  box-shadow:0 0 0 0 rgba(150,192,206,.7); animation:pulse 1.8s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(150,192,206,.6);}
  70%{box-shadow:0 0 0 10px rgba(150,192,206,0);} 100%{box-shadow:0 0 0 0 rgba(150,192,206,0);} }
.skyline { position:absolute; left:0; right:0; bottom:0; width:100%; height:50px; z-index:1; opacity:.28; }
.bigtaxi { position:absolute; right:26px; top:14px; z-index:2; opacity:.9; }

/* METRIC CARDS */
.mcard { background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:18px 20px;
  height:100%; box-shadow:0 10px 24px -18px rgba(15,65,74,.45); }
.mtop { display:flex; align-items:center; gap:9px; color:var(--muted); font-size:.74rem;
  font-weight:700; text-transform:uppercase; letter-spacing:.5px; }
.mval { font-family:'Fraunces',serif; font-weight:600; font-size:2rem; color:var(--midnight);
  margin-top:8px; line-height:1.05; }
.msub { font-size:.8rem; color:var(--muted); margin-top:2px; font-weight:500; }
.ic { width:18px; height:18px; fill:none; stroke-width:2; }

/* SECTIONS */
.sect { font-family:'Fraunces',serif; font-weight:600; font-size:1.25rem; color:var(--midnight);
  margin:8px 0 12px; display:flex; align-items:center; gap:10px; }

/* INSIGHT */
.insight { background:#FBF6EE; border:1px solid #E7D6C6; border-left:5px solid var(--maroon);
  border-radius:18px; padding:17px 20px; font-size:.98rem; color:#4A3A33; line-height:1.55; font-weight:500; }
.insight b { color:var(--maroon); }

/* RECEIPT */
.receipt { background:var(--surface);
  border:1px dashed var(--tan); border-radius:16px; padding:16px 18px;
  box-shadow:0 10px 24px -18px rgba(15,65,74,.4); font-size:.86rem; }
.r-h { font-family:'Fraunces',serif; font-weight:600; font-size:1rem; color:var(--midnight);
  display:flex; align-items:center; gap:8px; border-bottom:1px dashed var(--line); padding-bottom:8px; margin-bottom:8px; }
.r-row { display:flex; justify-content:space-between; padding:3px 0; color:var(--muted); }
.r-row b { color:var(--ink); font-weight:600; }
.r-total { border-top:1px dashed var(--line); margin-top:8px; padding-top:8px;
  display:flex; justify-content:space-between; font-family:'Fraunces',serif; font-weight:600;
  font-size:1.05rem; color:var(--maroon); }

/* ZONE CARDS */
.zcard { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:13px 15px;
  box-shadow:0 10px 24px -19px rgba(15,65,74,.45); margin-bottom:9px; }
.zrank { font-family:'Fraunces',serif; font-weight:600; font-size:.95rem; color:var(--maroon); }
.zname { font-weight:700; font-size:.93rem; margin:1px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.zboro { font-size:.74rem; color:var(--muted); font-weight:500; }
.zval { font-family:'Fraunces',serif; font-weight:600; font-size:1.35rem; margin-top:6px; color:var(--midnight); }
.zbar { height:6px; border-radius:100px; background:#EAE0D2; margin-top:7px; overflow:hidden; }
.zfill { height:100%; border-radius:100px; background:linear-gradient(90deg,var(--blue),var(--maroon)); }

/* DRIVER TIP */
.tip { background:#E8F0F2; border:1px solid #CFE0E5; border-radius:18px; padding:16px 19px;
  font-size:.95rem; color:#2C4248; font-weight:500; line-height:1.5; display:flex; gap:12px; align-items:flex-start; }

/* SIDEBAR */
section[data-testid="stSidebar"] { background:#FBF6EE; border-right:1px solid var(--line); }
.side-h { font-family:'Fraunces',serif; font-weight:600; font-size:1.08rem; color:var(--midnight); margin-bottom:8px; }
.side-cap { font-size:.82rem; color:var(--muted); margin-top:-2px; }
.side-foot { font-size:.74rem; color:var(--muted); margin-top:18px; line-height:1.5;
  border-top:1px solid var(--line); padding-top:12px; }
footer, #MainMenu { visibility:hidden; }
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

BIGTAXI = ('<svg class="bigtaxi" width="118" height="70" viewBox="0 0 120 72" fill="none">'
           '<path d="M14 44l6-19a10 10 0 019.6-7.2h40.8A10 10 0 0180 25l6 19" fill="#F2C200"/>'
           '<rect x="8" y="42" width="92" height="20" rx="6" fill="#F2C200"/>'
           '<rect x="40" y="13" width="26" height="9" rx="2" fill="#0F414A"/>'
           '<rect x="30" y="27" width="20" height="13" rx="2" fill="#EFE8DF"/>'
           '<rect x="56" y="27" width="22" height="13" rx="2" fill="#EFE8DF"/>'
           '<rect x="8" y="50" width="92" height="6" fill="#0F414A"/>'
           '<rect x="14" y="50" width="8" height="6" fill="#F2C200"/><rect x="30" y="50" width="8" height="6" fill="#F2C200"/>'
           '<rect x="46" y="50" width="8" height="6" fill="#F2C200"/><rect x="62" y="50" width="8" height="6" fill="#F2C200"/>'
           '<rect x="78" y="50" width="8" height="6" fill="#F2C200"/>'
           '<circle cx="30" cy="62" r="8" fill="#0F414A"/><circle cx="30" cy="62" r="3.4" fill="#EFE8DF"/>'
           '<circle cx="84" cy="62" r="8" fill="#0F414A"/><circle cx="84" cy="62" r="3.4" fill="#EFE8DF"/></svg>')

SKYLINE = ('<svg class="skyline" viewBox="0 0 1200 80" preserveAspectRatio="none" fill="#EFE8DF">'
           '<rect x="40" y="40" width="40" height="40"/><rect x="90" y="22" width="34" height="58"/>'
           '<rect x="132" y="48" width="44" height="32"/><rect x="250" y="30" width="30" height="50"/>'
           '<rect x="286" y="14" width="20" height="66"/><rect x="312" y="44" width="40" height="36"/>'
           '<rect x="470" y="36" width="36" height="44"/><rect x="512" y="8" width="24" height="72"/>'
           '<rect x="542" y="40" width="34" height="40"/><rect x="690" y="46" width="40" height="34"/>'
           '<rect x="736" y="26" width="28" height="54"/><rect x="900" y="38" width="34" height="42"/>'
           '<rect x="940" y="18" width="22" height="62"/><rect x="968" y="48" width="46" height="32"/>'
           '<rect x="1090" y="34" width="36" height="46"/></svg>')

IC = {
 "car":'<svg class="ic" viewBox="0 0 24 24" stroke="#7F0303"><path d="M5 13l1.4-4.2A2 2 0 018.3 7.5h7.4a2 2 0 011.9 1.3L19 13"/><rect x="4" y="13" width="16" height="5" rx="1.4"/><circle cx="7.5" cy="18" r="1.3"/><circle cx="16.5" cy="18" r="1.3"/></svg>',
 "flame":'<svg class="ic" viewBox="0 0 24 24" stroke="#7F0303"><path d="M12 3c1 3 4 4 4 8a4 4 0 01-8 0c0-1.6.6-2.6 1.1-3.1.2 1 .8 1.6 1.6 1.6C11 8 10 6 12 3z"/></svg>',
 "grid":'<svg class="ic" viewBox="0 0 24 24" stroke="#96C0CE"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></svg>',
 "clock":'<svg class="ic" viewBox="0 0 24 24" stroke="#0F414A"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
 "map":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7F0303" stroke-width="2"><path d="M9 5l6 2 6-2v12l-6 2-6-2-6 2V7z"/><path d="M9 5v12M15 7v12"/></svg>',
 "pin":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7F0303" stroke-width="2"><path d="M12 21s-6-5.3-6-10a6 6 0 1112 0c0 4.7-6 10-6 10z"/><circle cx="12" cy="11" r="2"/></svg>',
 "pulse":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0F414A" stroke-width="2"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>',
 "bulb":'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#155A63" stroke-width="2"><path d="M9 18h6M10 21h4M12 3a6 6 0 00-4 10.5c.7.7 1 1.3 1 2.5h6c0-1.2.3-1.8 1-2.5A6 6 0 0012 3z"/></svg>',
 "receipt":'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0F414A" stroke-width="2"><path d="M6 3v18l2-1 2 1 2-1 2 1 2-1 2 1V3l-2 1-2-1-2 1-2-1-2 1z"/><path d="M9 8h6M9 12h6"/></svg>',
}

# HERO
st.markdown(f"""
<div class="hero rise"><div class="hero-row">
  <div class="hero-mark"><svg width="30" height="30" viewBox="0 0 24 24" fill="#F2C200">
    <path d="M3.5 13l1.3-4A2.3 2.3 0 017 7.4h10a2.3 2.3 0 012.2 1.6l1.3 4"/>
    <rect x="2.6" y="12.6" width="18.8" height="5.4" rx="1.6"/>
    <rect x="9" y="5.6" width="6" height="2.1" rx=".5"/>
    <circle cx="7.2" cy="18.4" r="1.7" fill="#0F414A"/><circle cx="16.8" cy="18.4" r="1.7" fill="#0F414A"/></svg></div>
  <div><h1>NYC Fleet Command Center</h1>
  <p>Gett · predictive taxi-demand across New York City</p></div>
  <div class="pill"><span class="dot"></span>FORECAST LIVE</div>
</div>{BIGTAXI}{SKYLINE}</div>
<div class="taxi-stripe"></div>
""", unsafe_allow_html=True)

# METRICS
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
                 f'<div class="mval">{val}</div><div class="msub">{sub}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([2.15, 1])

with left:
    st.markdown(f'<div class="sect">{IC["map"]} Demand map · {fmt_hour(hour)}, {day_name}</div>', unsafe_allow_html=True)
    scale = [[0.0,"#DCEBF0"],[0.22,"#96C0CE"],[0.45,"#D8BA98"],
             [0.66,"#D98E63"],[0.84,"#B23A1E"],[1.0,"#7F0303"]]
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson, locations=cur["PULocationID"], z=cur["predicted_demand"],
        featureidkey="id", colorscale=scale, marker_opacity=0.86,
        marker_line_width=0.5, marker_line_color="rgba(239,232,223,0.7)",
        colorbar=dict(title="rides/hr", thickness=13, len=0.82,
                      tickfont=dict(family="Plus Jakarta Sans", size=11, color="#7C8A8E")),
        customdata=np.stack([cur["zone"], cur["borough"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Demand: %{z:.0f} rides/hr<extra></extra>"))
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=9.3,
        mapbox_center={"lat":40.730,"lon":-73.945}, margin=dict(l=0,r=0,t=0,b=0),
        height=540, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True,
        config={"displayModeBar":True,"displaylogo":False,"modeBarButtonsToRemove":["lasso2d","select2d"]})

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
    # Receipt-style summary card
    st.markdown(
        f'<div class="receipt rise">'
        f'<div class="r-h">{IC["receipt"]} Fleet ticket</div>'
        f'<div class="r-row">Date <b>{picked_date.strftime("%b %d, %Y")}</b></div>'
        f'<div class="r-row">Day <b>{day_name}</b></div>'
        f'<div class="r-row">Time <b>{fmt_hour(hour)}</b></div>'
        f'<div class="r-row">Hottest <b>{ZONE_NAME.get(int(hottest["PULocationID"]),"")}</b></div>'
        f'<div class="r-row">Busy zones <b>{busy_zones}</b></div>'
        f'<div class="r-total"><span>Total demand</span><span>{total_demand:,}</span></div>'
        f'</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="sect">{IC["pin"]} Top 5 hottest zones</div>', unsafe_allow_html=True)
    top5 = top.head(5)
    maxv = top5["predicted_demand"].max()
    for i, row in top5.iterrows():
        lid = int(row["PULocationID"])
        fill = row["predicted_demand"]/maxv*100
        st.markdown(
            f'<div class="zcard rise"><div class="zrank">#{i+1}</div>'
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

# --------------------------------------------------------------------------- #
# PULSE OF THE DAY  — animated (built-in Plotly Play button, smooth)
# --------------------------------------------------------------------------- #
st.write("")
st.markdown(f'<div class="sect">{IC["pulse"]} Pulse of the day · {day_name} '
            '<span style="font-size:.8rem;color:#7C8A8E;font-family:Plus Jakarta Sans;font-weight:500;">'
            '— press ▶ to watch the city wake up</span></div>', unsafe_allow_html=True)

xs = list(range(24))
ys = list(day_curve.values)
base_line = go.Scatter(x=xs, y=ys, mode="lines",
                       line=dict(color="#7F0303", width=3.5, shape="spline"),
                       fill="tozeroy", fillcolor="rgba(127,3,3,0.10)",
                       hovertemplate="%{x}:00 — %{y:,.0f} rides<extra></extra>")
frames = [go.Frame(name=str(h), data=[base_line,
            go.Scatter(x=[h], y=[ys[h]], mode="markers+text",
                       marker=dict(color="#0F414A", size=17, line=dict(color="#EFE8DF", width=3)),
                       text=[fmt_hour(h)], textposition="top center",
                       textfont=dict(family="Plus Jakarta Sans", size=12, color="#0F414A"),
                       hoverinfo="skip")]) for h in xs]

anim = go.Figure(
    data=[base_line,
          go.Scatter(x=[hour], y=[ys[hour]], mode="markers+text",
                     marker=dict(color="#0F414A", size=17, line=dict(color="#EFE8DF", width=3)),
                     text=[fmt_hour(hour)], textposition="top center",
                     textfont=dict(family="Plus Jakarta Sans", size=12, color="#0F414A"),
                     hoverinfo="skip")],
    frames=frames)
anim.update_layout(
    height=300, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 3)),
               ticktext=[fmt_hour(h) for h in range(0, 24, 3)], showgrid=False,
               tickfont=dict(family="Plus Jakarta Sans", size=11, color="#7C8A8E")),
    yaxis=dict(showgrid=True, gridcolor="#E7DFD2", zeroline=False,
               tickfont=dict(family="Plus Jakarta Sans", size=11, color="#7C8A8E")),
    updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.18, xanchor="left",
        bgcolor="#0F414A", bordercolor="#0F414A", font=dict(color="#EFE8DF", size=12),
        buttons=[
            dict(label="▶  Play the day", method="animate",
                 args=[None, dict(frame=dict(duration=140, redraw=True),
                                  fromcurrent=True, transition=dict(duration=90, easing="cubic-in-out"))]),
            dict(label="❚❚  Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ])])
st.plotly_chart(anim, use_container_width=True, config={"displayModeBar": False})
