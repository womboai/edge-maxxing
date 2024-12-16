import json
from operator import itemgetter
from pathlib import Path
from statistics import mean

import numpy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
from scipy.stats import percentileofscore

TEST_DATA_DIRECTORY = Path(__file__).parent.parent / "test_data"

days = [file.stem for file in TEST_DATA_DIRECTORY.glob("*.json")]

app = Dash()

def visualize_validator(data: dict, percentile: float):
    values = sorted(((hotkey, score_info["score"]) for hotkey, score_info in data.items()), key=itemgetter(1))

    scores = [
        (
            f"{values[i][0][:5]}..",
            values[i][1],
            0.0 if i == 0 else values[i][1] - values[i - 1][1],
        ) for i in range(len(values))
    ]

    deviations = numpy.array([
        values[i + 1][1] - values[i][1]
        for i in range(len(values) - 1)
        if values[i][1] > 0
    ])

    q1 = numpy.percentile(deviations, 25)
    q3 = numpy.percentile(deviations, 75)
    iqr = q3 - q1

    anomaly_threshold = q3 + iqr * 1.5

    mean_threshold = mean(deviations)

    data_frame = pd.DataFrame(
        {
            "hotkey": list(map(itemgetter(0), scores)),
            "score": list(map(itemgetter(1), scores)),
            "difference_percentile": [percentileofscore(deviations, score_data[2]) for score_data in scores],
            "difference": list(map(itemgetter(2), scores)),
        }
    )

    figure = px.bar(data_frame, x="hotkey", y="difference", color="difference_percentile")

    if percentile:
        threshold = numpy.percentile(deviations, percentile)
        figure.add_traces(go.Scatter(x=data_frame.hotkey, y=len(data_frame) * [threshold], mode="lines"))

    figure.add_traces(go.Scatter(x=data_frame.hotkey, y=len(data_frame) * [mean_threshold], mode="lines"))
    figure.add_traces(go.Scatter(x=data_frame.hotkey, y=len(data_frame) * [anomaly_threshold], mode="lines"))

    return dcc.Graph(figure=figure)

@app.callback(
    Output("data-slot", "children"),
    [
        Input("day-filter", "value"),
        Input("percentile", "value"),
    ],
)
def callback_func(day: str, percentile: float):
    with open(TEST_DATA_DIRECTORY / f"{day}.json") as f:
        data = json.load(f)

    return [
        html.Div(
            children=[
                html.Label(str(uid)),
                visualize_validator(validator_data, percentile),
            ]
        )
        for uid, validator_data in data.items()
    ]

app.layout = html.Div(children=[
    dcc.Dropdown(days, days[0], id="day-filter"),
    dcc.Slider(0, 100, value=90, id="percentile"),
    html.Div(id="data-slot"),
])

app.run()
