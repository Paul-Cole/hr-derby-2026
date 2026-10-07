"""2026 fantasy HR derby — live leaderboard since August 14."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, dash_table, dcc, html

START = date(2026, 8, 14)
SEASON = START.year
OUT_DIR = Path(__file__).resolve().parent
TRACK_DIR = Path(os.environ.get("TRACK_DIR", str(OUT_DIR / "fantasy_2026_hr")))
TRACK_DIR.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("PORT", "8051"))

ROSTERS = [
    ("Paul", "C", "Shea Langeliers", 669127),
    ("Paul", "1B", "Munetaka Murakami", 808959),
    ("Paul", "2B", "Ketel Marte", 606466),
    ("Paul", "SS", "Elly De La Cruz", 682829),
    ("Paul", "3B", "Miguel Vargas", 678246),
    ("Paul", "LF", "Riley Greene", 682985),
    ("Paul", "CF", "Byron Buxton", 621439),
    ("Paul", "RF", "Wilyer Abreu", 677800),
    ("Paul", "DH", "Yordan Alvarez", 670541),
    ("Ty", "C", "Hunter Goodman", 696100),
    ("Ty", "1B", "Willson Contreras", 575929),
    ("Ty", "2B", "Brandon Lowe", 664040),
    ("Ty", "SS", "CJ Abrams", 682928),
    ("Ty", "3B", "Alex Bregman", 608324),
    ("Ty", "LF", "Juan Soto", 665742),
    ("Ty", "CF", "Jackson Merrill", 701538),
    ("Ty", "RF", "Ronald Acuña Jr.", 660670),
    ("Ty", "DH", "Shohei Ohtani", 660271),
    ("Drake", "C", "Liam Hicks", 689414),
    ("Drake", "1B", "Matt Olson", 621566),
    ("Drake", "2B", "Curtis Mead", 678554),
    ("Drake", "SS", "Colson Montgomery", 695657),
    ("Drake", "3B", "Manny Machado", 592518),
    ("Drake", "LF", "Ian Happ", 664023),
    ("Drake", "CF", "Andy Pages", 681624),
    ("Drake", "RF", "James Wood", 695578),
    ("Drake", "DH", "Ben Rice", 700250),
    ("Nick", "C", "Cal Raleigh", 663728),
    ("Nick", "1B", "Pete Alonso", 624413),
    ("Nick", "2B", "Jazz Chisholm Jr.", 665862),
    ("Nick", "SS", "Bobby Witt Jr.", 677951),
    ("Nick", "3B", "José Ramírez", 608070),
    ("Nick", "LF", "Corbin Carroll", 682998),
    ("Nick", "CF", "Mike Trout", 545361),
    ("Nick", "RF", "Bryce Harper", 547180),
    ("Nick", "DH", "Kyle Schwarber", 656941),
    ("Trey", "C", "Drake Baldwin", 686948),
    ("Trey", "1B", "Freddie Freeman", 518692),
    ("Trey", "2B", "Ozzie Albies", 645277),
    ("Trey", "SS", "Mookie Betts", 605141),
    ("Trey", "3B", "Junior Caminero", 691406),
    ("Trey", "LF", "Teoscar Hernández", 606192),
    ("Trey", "CF", "Pete Crow-Armstrong", 691718),
    ("Trey", "RF", "Seiya Suzuki", 673548),
    ("Trey", "DH", "Joc Pederson", 592626),
]

OWNERS = ["Paul", "Ty", "Drake", "Nick", "Trey"]
TEAM_COLORS = {
    "Paul": "#3b82f6",
    "Ty": "#22c55e",
    "Drake": "#f59e0b",
    "Nick": "#a855f7",
    "Trey": "#ef4444",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-fantasy-hr-derby/2026"})

# statsapi gameLog defaults to regular season only, so October games vanish from
# the log unless every postseason round is asked for by name:
# R regular, F wild card, D division, L championship, W World Series.
GAME_TYPES = "R,F,D,L,W"

BG = "#0b1220"
CARD = "#121a2b"
TEXT = "#e8eef9"
MUTED = "#93a0b8"
LINE = "#243049"

roster_df = pd.DataFrame(ROSTERS, columns=["owner", "slot", "player", "player_id"])


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def fetch_game_log(player_id: int) -> list[dict]:
    url = f"https://statsapi.mlb.com/api/v1/people/{int(player_id)}/stats"
    resp = SESSION.get(
        url,
        params={
            "stats": "gameLog",
            "group": "hitting",
            "season": SEASON,
            "gameType": GAME_TYPES,
        },
        timeout=30,
    )
    resp.raise_for_status()
    rows = []
    for block in resp.json().get("stats") or []:
        for split in block.get("splits", []):
            day = _parse_day(split.get("date"))
            if day is None or day < START:
                continue
            rows.append(
                {
                    "player_id": int(player_id),
                    "date": day,
                    "hr": int(split.get("stat", {}).get("homeRuns") or 0),
                }
            )
    return rows


def _boxscore_hrs(game_pk: int, player_ids: set[int]) -> dict[int, int]:
    resp = SESSION.get(
        f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore",
        timeout=30,
    )
    resp.raise_for_status()
    found: dict[int, int] = {}
    teams = resp.json().get("teams") or {}
    for side in ("home", "away"):
        players = (teams.get(side) or {}).get("players") or {}
        for key, pdata in players.items():
            pid = int((pdata.get("person") or {}).get("id") or str(key).replace("ID", ""))
            if pid not in player_ids:
                continue
            batting = (pdata.get("stats") or {}).get("batting") or {}
            hr = int(batting.get("homeRuns") or 0)
            if hr:
                found[pid] = found.get(pid, 0) + hr
    return found


def fetch_today_boxscore_hr(player_ids: set[int], day: date) -> dict[int, int]:
    """HR from today's boxscores, including in-progress games."""
    resp = SESSION.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": day.isoformat()},
        timeout=30,
    )
    resp.raise_for_status()
    pks = []
    for day_block in resp.json().get("dates", []):
        for game in day_block.get("games", []):
            state = (game.get("status") or {}).get("abstractGameState")
            if state in {"Live", "Final"}:
                pks.append(game["gamePk"])
    out: dict[int, int] = {pid: 0 for pid in player_ids}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_boxscore_hrs, pk, player_ids) for pk in pks]
        for fut in as_completed(futs):
            try:
                for pid, hr in fut.result().items():
                    out[pid] += hr
            except Exception as exc:
                print(f"boxscore failed: {exc}")
    return out


def load_hr_data() -> dict:
    today = datetime.now().date()
    ids = list(roster_df["player_id"])
    log_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(fetch_game_log, pid): pid for pid in ids}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                log_rows.extend(fut.result())
            except Exception as exc:
                print(f"game log failed for {pid}: {exc}")

    logs = pd.DataFrame(log_rows)
    if logs.empty:
        logs = pd.DataFrame(columns=["player_id", "date", "hr"])

    try:
        live = fetch_today_boxscore_hr(set(ids), today)
    except Exception as exc:
        print(f"today boxscore failed: {exc}")
        live = {}

    if not logs.empty:
        # one row per player-day; a doubleheader arrives as two splits
        logs = logs.groupby(["player_id", "date"], as_index=False)["hr"].sum()

    if live:
        live_df = pd.DataFrame(
            [{"player_id": pid, "date": today, "hr": hr} for pid, hr in live.items() if hr]
        )
        if not live_df.empty:
            # Today can come from both sources; keep the higher count. Earlier days
            # stay summed so doubleheaders aren't flattened to a single game.
            rest = logs[logs["date"] != today]
            merged_today = (
                pd.concat([logs[logs["date"] == today], live_df], ignore_index=True)
                .groupby(["player_id", "date"], as_index=False)["hr"]
                .max()
            )
            logs = pd.concat([rest, merged_today], ignore_index=True)

    players = roster_df.merge(logs, on="player_id", how="left")
    players["date"] = pd.to_datetime(players["date"]).dt.date
    players["hr"] = players["hr"].fillna(0).astype(int)

    end = max(today, START)
    all_days = pd.date_range(START, end, freq="D").date
    grid = (
        roster_df[["owner", "slot", "player", "player_id"]]
        .assign(key=1)
        .merge(pd.DataFrame({"date": all_days, "key": 1}), on="key")
        .drop(columns="key")
    )
    daily = grid.merge(
        players[["player_id", "date", "hr"]],
        on=["player_id", "date"],
        how="left",
    )
    daily["hr"] = daily["hr"].fillna(0).astype(int)

    team_daily = daily.groupby(["date", "owner"], as_index=False)["hr"].sum()
    team_daily = team_daily.sort_values(["owner", "date"])
    team_daily["cumulative_hr"] = team_daily.groupby("owner")["hr"].cumsum()

    current = (
        team_daily.sort_values("date")
        .groupby("owner", as_index=False)
        .tail(1)[["owner", "cumulative_hr"]]
        .rename(columns={"cumulative_hr": "hr"})
    )
    today_hr = team_daily[team_daily["date"] == today][["owner", "hr"]].rename(
        columns={"hr": "today_hr"}
    )
    board = current.merge(today_hr, on="owner", how="left")
    board["today_hr"] = board["today_hr"].fillna(0).astype(int)
    board = board.set_index("owner").reindex(OWNERS).reset_index()
    board["hr"] = board["hr"].fillna(0).astype(int)
    board = board.sort_values(["hr", "today_hr"], ascending=False).reset_index(drop=True)
    board["rank"] = range(1, len(board) + 1)

    player_tot = (
        daily.groupby(["owner", "slot", "player", "player_id"], as_index=False)["hr"]
        .sum()
        .sort_values(["hr", "player"], ascending=[False, True])
    )
    player_today = daily[daily["date"] == today][["player_id", "hr"]].rename(
        columns={"hr": "today_hr"}
    )
    player_tot = player_tot.merge(player_today, on="player_id", how="left")
    player_tot["today_hr"] = player_tot["today_hr"].fillna(0).astype(int)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        player_tot.assign(as_of=stamp).to_csv(
            TRACK_DIR / f"snapshot_{today.isoformat()}.csv", index=False
        )
        team_daily.to_csv(TRACK_DIR / "team_daily.csv", index=False)
    except OSError as exc:
        print(f"snapshot write skipped: {exc}")

    return {
        "as_of": stamp,
        "today": today.isoformat(),
        "board": board,
        "team_daily": team_daily,
        "players": player_tot,
        "daily": daily,
    }


def line_figure(team_daily: pd.DataFrame) -> go.Figure:
    fig = px.line(
        team_daily,
        x="date",
        y="cumulative_hr",
        color="owner",
        color_discrete_map=TEAM_COLORS,
        markers=True,
        labels={"date": "Date", "cumulative_hr": "Home runs since Aug 14", "owner": "Team"},
        title="Team home runs since August 14, 2026",
        category_orders={"owner": OWNERS},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Georgia, serif"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=40, r=20, t=80, b=40),
        hovermode="x unified",
        xaxis=dict(gridcolor=LINE),
        yaxis=dict(gridcolor=LINE, rangemode="tozero", title="Cumulative HR"),
    )
    return fig


def bar_figure(board: pd.DataFrame) -> go.Figure:
    ordered = board.sort_values("hr", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["hr"],
            y=ordered["owner"],
            orientation="h",
            marker_color=[TEAM_COLORS[o] for o in ordered["owner"]],
            text=ordered["hr"],
            textposition="outside",
            hovertemplate="%{y}: %{x} HR<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Georgia, serif"),
        title="Current leaderboard",
        margin=dict(l=80, r=40, t=60, b=40),
        xaxis=dict(title="HR since Aug 14", gridcolor=LINE, rangemode="tozero"),
        yaxis=dict(title=""),
        height=320,
    )
    return fig


def player_bar_figure(players: pd.DataFrame, owner: str) -> go.Figure:
    subset = players[players["owner"] == owner].copy()
    if subset.empty:
        subset = pd.DataFrame(
            {"player": ["No players"], "slot": [""], "hr": [0], "today_hr": [0]}
        )
    subset["label"] = subset.apply(
        lambda r: f"{r['player']} ({r['slot']})" if r.get("slot") else r["player"],
        axis=1,
    )
    subset = subset.sort_values("hr", ascending=True)
    color = TEAM_COLORS.get(owner, "#3b82f6")
    fig = go.Figure(
        go.Bar(
            x=subset["hr"],
            y=subset["label"],
            orientation="h",
            marker_color=color,
            text=subset["hr"],
            textposition="outside",
            customdata=subset[["slot", "today_hr"]],
            hovertemplate="%{y}<br>Since Aug 14: %{x} HR<br>Today: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Georgia, serif"),
        title=f"{owner}'s roster — HR since August 14",
        margin=dict(l=160, r=40, t=60, b=40),
        xaxis=dict(title="HR since Aug 14", gridcolor=LINE, rangemode="tozero"),
        yaxis=dict(title=""),
        height=420,
    )
    return fig


def write_static_html(data: dict) -> Path:
    board = data["board"]
    players = data["players"]
    line = line_figure(data["team_daily"])
    bars = bar_figure(board)
    line_html = line.to_html(full_html=False, include_plotlyjs="cdn")
    bar_html = bars.to_html(full_html=False, include_plotlyjs=False)

    rows = "\n".join(
        f"<tr><td>{int(r.rank)}</td><td>{r.owner}</td>"
        f"<td>{int(r.hr)}</td><td>{int(r.today_hr)}</td></tr>"
        for r in board.itertuples()
    )
    player_rows = "\n".join(
        f"<tr><td>{r.owner}</td><td>{r.slot}</td><td>{r.player}</td>"
        f"<td>{int(r.hr)}</td><td>{int(r.today_hr)}</td></tr>"
        for r in players.itertuples()
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>2026 HR Derby — since Aug 14</title>
  <style>
    body {{ background:{BG}; color:{TEXT}; font-family: Georgia, serif; margin:0; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
    h1 {{ font-size: 32px; margin: 0 0 8px; }}
    .sub {{ color:{MUTED}; margin-bottom: 28px; }}
    table {{ width:100%; border-collapse: collapse; background:{CARD}; }}
    th, td {{ padding: 10px 12px; text-align:left; border-bottom: 1px solid {LINE}; }}
    th {{ color:{MUTED}; font-weight: 600; font-size: 13px; }}
    .grid {{ display:grid; gap:24px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
  </style>
</head>
<body>
<main>
  <h1>2026 HR Derby</h1>
  <p class="sub">Home runs since August 14, 2026 · updated {data["as_of"]}</p>
  <h2>Leaderboard</h2>
  <table>
    <thead><tr><th>Rank</th><th>Team</th><th>HR since Aug 14</th><th>Today</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {bar_html}
  {line_html}
  <h2>Picks</h2>
  <table>
    <thead><tr><th>Team</th><th>Slot</th><th>Player</th><th>HR since Aug 14</th><th>Today</th></tr></thead>
    <tbody>{player_rows}</tbody>
  </table>
</main>
</body>
</html>
"""
    path = OUT_DIR / "fantasy_2026.html"
    path.write_text(html_doc, encoding="utf-8")
    return path


TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": CARD,
        "color": MUTED,
        "fontWeight": "600",
        "border": f"1px solid {LINE}",
        "fontFamily": "Georgia, serif",
    },
    "style_cell": {
        "backgroundColor": CARD,
        "color": TEXT,
        "border": f"1px solid {LINE}",
        "textAlign": "left",
        "padding": "10px 12px",
        "fontFamily": "Georgia, serif",
        "fontSize": 15,
    },
    "style_data_conditional": [
        {"if": {"row_index": "odd"}, "backgroundColor": "#0f1726"}
    ],
}

app = Dash(__name__, title="2026 HR Derby")
app.layout = html.Div(
    style={
        "backgroundColor": BG,
        "color": TEXT,
        "minHeight": "100vh",
        "fontFamily": "Georgia, serif",
        "padding": "32px 24px 64px",
    },
    children=[
        html.Div(
            style={"maxWidth": 1100, "margin": "0 auto"},
            children=[
                html.H1("2026 HR Derby", style={"margin": "0 0 6px"}),
                html.Div(
                    "Team home runs since August 14, 2026 · live MLB game logs",
                    style={"color": MUTED, "marginBottom": 18},
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "gap": 12,
                        "alignItems": "center",
                        "marginBottom": 24,
                    },
                    children=[
                        html.Button(
                            "Refresh now",
                            id="refresh",
                            n_clicks=0,
                            style={
                                "background": "#2563eb",
                                "color": "white",
                                "border": 0,
                                "padding": "8px 14px",
                                "borderRadius": 6,
                                "cursor": "pointer",
                                "fontFamily": "Georgia, serif",
                            },
                        ),
                        html.Span(id="updated", style={"color": MUTED}),
                    ],
                ),
                html.Div(id="rank-cards"),
                html.H2("Leaderboard", style={"marginTop": 28}),
                dash_table.DataTable(
                    id="board-table",
                    columns=[
                        {"name": "Rank", "id": "rank"},
                        {"name": "Team", "id": "owner"},
                        {"name": "HR since Aug 14", "id": "hr"},
                        {"name": "Today", "id": "today_hr"},
                    ],
                    **TABLE_STYLE,
                ),
                dcc.Graph(id="bar-chart", config={"displayModeBar": False}),
                dcc.Graph(id="line-chart", config={"displayModeBar": False}),
                html.H2("Players by team", style={"marginTop": 28}),
                html.Div(
                    "Choose a manager to see which of their picks are producing.",
                    style={"color": MUTED, "marginBottom": 10},
                ),
                dcc.Dropdown(
                    id="team-dropdown",
                    options=[{"label": name, "value": name} for name in OWNERS],
                    value="Paul",
                    clearable=False,
                    style={
                        "width": 280,
                        "marginBottom": 12,
                        "color": "#111",
                    },
                ),
                dcc.Graph(id="player-bar", config={"displayModeBar": False}),
                html.H2("Picks"),
                dash_table.DataTable(
                    id="player-table",
                    columns=[
                        {"name": "Team", "id": "owner"},
                        {"name": "Slot", "id": "slot"},
                        {"name": "Player", "id": "player"},
                        {"name": "HR since Aug 14", "id": "hr"},
                        {"name": "Today", "id": "today_hr"},
                    ],
                    sort_action="native",
                    **TABLE_STYLE,
                ),
                dcc.Store(id="data-store"),
                dcc.Interval(id="tick", interval=5 * 60 * 1000, n_intervals=0),
            ],
        )
    ],
)


@app.callback(
    Output("updated", "children"),
    Output("rank-cards", "children"),
    Output("board-table", "data"),
    Output("player-table", "data"),
    Output("bar-chart", "figure"),
    Output("line-chart", "figure"),
    Output("data-store", "data"),
    Input("refresh", "n_clicks"),
    Input("tick", "n_intervals"),
)
def refresh(_clicks, _ticks):
    data = load_hr_data()
    try:
        write_static_html(data)
    except OSError as exc:
        print(f"html write skipped: {exc}")
    board = data["board"]
    cards = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({len(board)}, minmax(0, 1fr))",
            "gap": 12,
        },
        children=[
            html.Div(
                style={
                    "background": CARD,
                    "border": f"1px solid {LINE}",
                    "borderTop": f"3px solid {TEAM_COLORS[row.owner]}",
                    "padding": "14px 16px",
                },
                children=[
                    html.Div(
                        f"#{int(row.rank)} {row.owner}",
                        style={"color": MUTED, "fontSize": 13},
                    ),
                    html.Div(
                        str(int(row.hr)),
                        style={"fontSize": 32, "lineHeight": "1.1", "marginTop": 4},
                    ),
                    html.Div(
                        f"+{int(row.today_hr)} today",
                        style={"color": MUTED, "marginTop": 4},
                    ),
                ],
            )
            for row in board.itertuples()
        ],
    )
    team_daily = data["team_daily"].copy()
    team_daily["date"] = team_daily["date"].astype(str)
    store = {
        "players": data["players"][
            ["owner", "slot", "player", "hr", "today_hr"]
        ].to_dict("records"),
        "team_daily": team_daily.to_dict("records"),
    }
    return (
        f"Updated {data['as_of']}",
        cards,
        board.to_dict("records"),
        store["players"],
        bar_figure(board),
        line_figure(data["team_daily"]),
        store,
    )


@app.callback(
    Output("player-bar", "figure"),
    Input("team-dropdown", "value"),
    Input("data-store", "data"),
)
def update_player_bar(owner, store):
    owner = owner or "Paul"
    if not store:
        return player_bar_figure(pd.DataFrame(columns=["owner", "slot", "player", "hr", "today_hr"]), owner)
    players = pd.DataFrame(store.get("players") or [])
    return player_bar_figure(players, owner)


server = app.server

if __name__ == "__main__":
    print(f"HR Derby at http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
