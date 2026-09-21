from pathlib import Path

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go


CSV_PATH = Path("2022_2022-08-10_Transect_Aug-10th-2022-11-46AM-Flight-Airdata.csv")

REQUIRED_COLUMNS = [
    "time(millisecond)",
    "datetime(utc)",
    "latitude",
    "longitude",
    "height_above_takeoff(feet)",
    "height_above_ground_at_drone_location(feet)",
    "ground_elevation_at_drone_location(feet)",
    "altitude_above_seaLevel(feet)",
    "height_sonar(feet)",
]


def load_flight(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    # Remove rows where GPS position isn't available.
    df = df.dropna(subset=[
        "time(millisecond)",
        "latitude",
        "longitude"
    ]).copy()

    df["time(millisecond)"] = pd.to_numeric(
        df["time(millisecond)"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=[
        "time(millisecond)",
        "latitude",
        "longitude"
    ])

    df = df.sort_values("time(millisecond)").reset_index(drop=True)

    # Normalize playback so the flight always starts at 0.
    df["playback_ms"] = (
        df["time(millisecond)"] - df["time(millisecond)"].iloc[0]
    )

    return df


df = load_flight(CSV_PATH)

MAX_TIME = int(df["playback_ms"].max())


def get_current_row(playback_ms):
    """
    Return the row whose timestamp is closest to the
    current playback time.
    """

    index = (df["playback_ms"] - playback_ms).abs().idxmin()

    return df.loc[index]


def create_figure(playback_ms, markers):
    current = get_current_row(playback_ms)

    fig = go.Figure()

    # Entire flight path
    fig.add_trace(
        go.Scattermap(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="lines",
            line=dict(width=3),
            name="Flight path",
            hoverinfo="skip",
        )
    )

    # Current drone position
    fig.add_trace(
        go.Scattermap(
            lat=[current["latitude"]],
            lon=[current["longitude"]],
            mode="markers",
            marker=dict(size=16),
            name="Drone",
            customdata=[[
                current["datetime(utc)"],
                current["height_above_takeoff(feet)"],
                current["height_above_ground_at_drone_location(feet)"],
                current["altitude_above_seaLevel(feet)"],
            ]],
            hovertemplate=(
                "<b>Drone</b><br>"
                "Latitude: %{lat:.6f}<br>"
                "Longitude: %{lon:.6f}<br>"
                "UTC: %{customdata[0]}<br>"
                "Height above takeoff: %{customdata[1]:.1f} ft<br>"
                "Height above ground: %{customdata[2]:.1f} ft<br>"
                "Altitude ASL: %{customdata[3]:.1f} ft"
                "<extra></extra>"
            ),
        )
    )

    # Persistent user-created markers
    if markers:
        fig.add_trace(
            go.Scattermap(
                lat=[m["lat"] for m in markers],
                lon=[m["lon"] for m in markers],
                mode="markers",
                marker=dict(size=12),
                text=[
                    f"Marker {i + 1}<br>{m['time'] / 1000:.2f}s"
                    for i, m in enumerate(markers)
                ],
                name="Markers",
            )
        )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(
                lat=current["latitude"],
                lon=current["longitude"],
            ),
            zoom=16,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
    )

    return fig


app = Dash(__name__)


app.layout = html.Div(
    [
        html.H2("Litchi Flight Playback"),

        dcc.Graph(
            id="flight-map",
            style={"height": "75vh"},
        ),

        html.Div(
            [
                html.Button(
                    "Play",
                    id="play-button",
                    n_clicks=0,
                ),

                html.Button(
                    "Add Marker",
                    id="marker-button",
                    n_clicks=0,
                    style={"marginLeft": "10px"},
                ),

                html.Span(
                    " Playback speed: ",
                    style={"marginLeft": "20px"},
                ),

                dcc.Dropdown(
                    id="speed",
                    options=[
                        {"label": "0.25×", "value": 0.25},
                        {"label": "0.5×", "value": 0.5},
                        {"label": "1×", "value": 1},
                        {"label": "2×", "value": 2},
                        {"label": "5×", "value": 5},
                        {"label": "10×", "value": 10},
                    ],
                    value=1,
                    clearable=False,
                    style={
                        "width": "100px",
                        "display": "inline-block",
                    },
                ),
            ],
            style={"marginBottom": "15px"},
        ),

        dcc.Slider(
            id="timeline",
            min=0,
            max=MAX_TIME,
            value=0,
            step=100,
            tooltip={
                "placement": "bottom",
                "always_visible": True,
            },
        ),

        html.Div(
            id="telemetry",
            style={
                "marginTop": "15px",
                "fontFamily": "monospace",
            },
        ),

        # Playback timer
        dcc.Interval(
            id="playback-interval",
            interval=100,
            disabled=True,
        ),

        # Stores whether we're playing
        dcc.Store(
            id="playing",
            data=False,
        ),

        # Persistent markers
        dcc.Store(
            id="markers",
            data=[],
        ),
    ]
)


@app.callback(
    Output("playing", "data"),
    Output("playback-interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("playing", "data"),
    prevent_initial_call=True,
)
def toggle_playback(n_clicks, playing):
    playing = not playing

    return (
        playing,
        not playing,
        "Pause" if playing else "Play",
    )


@app.callback(
    Output("timeline", "value"),
    Input("playback-interval", "n_intervals"),
    State("timeline", "value"),
    State("speed", "value"),
    State("playing", "data"),
)
def advance_playback(n_intervals, current_time, speed, playing):
    if not playing:
        return current_time

    # Interval fires every 100 ms.
    new_time = current_time + (100 * speed)

    if new_time >= MAX_TIME:
        return MAX_TIME

    return new_time


@app.callback(
    Output("markers", "data"),
    Input("marker-button", "n_clicks"),
    State("timeline", "value"),
    State("markers", "data"),
    prevent_initial_call=True,
)
def add_marker(n_clicks, playback_ms, markers):
    current = get_current_row(playback_ms)

    markers.append({
        "lat": current["latitude"],
        "lon": current["longitude"],
        "time": playback_ms,
    })

    return markers


@app.callback(
    Output("flight-map", "figure"),
    Output("telemetry", "children"),
    Input("timeline", "value"),
    Input("markers", "data"),
)
def update_display(playback_ms, markers):
    row = get_current_row(playback_ms)

    figure = create_figure(playback_ms, markers)

    seconds = playback_ms / 1000

    telemetry = (
        f"Time: {seconds:.2f}s | "
        f"Lat: {row['latitude']:.6f} | "
        f"Lon: {row['longitude']:.6f} | "
        f"Takeoff height: "
        f"{row['height_above_takeoff(feet)']:.1f} ft | "
        f"AGL: "
        f"{row['height_above_ground_at_drone_location(feet)']:.1f} ft"
    )

    return figure, telemetry


if __name__ == "__main__":
    app.run(debug=True)