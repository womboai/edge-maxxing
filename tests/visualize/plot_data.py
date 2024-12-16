import json
from operator import itemgetter
from pathlib import Path

import numpy
from dash import Dash, html, dcc, Output, Input
import pandas as pd
import plotly.express as px
from scipy.stats import percentileofscore

TEST_DATA_DIRECTORY = Path(__file__).parent.parent / "test_data"

days = [file.name[:-5] for file in TEST_DATA_DIRECTORY.iterdir()]

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

    score_differences_array = numpy.array(list(map(itemgetter(2), scores)))

    data_frame = pd.DataFrame(
        {
            "hotkey": list(map(itemgetter(0), scores)),
            "score": list(map(itemgetter(1), scores)),
            "difference_percentile": [percentileofscore(score_differences_array, score_difference) for score_difference in score_differences_array],
            "difference": score_differences_array,
        }
    )

    figure = px.bar(data_frame, x="hotkey", y="difference", color="difference_percentile")

    if percentile:
        threshold = numpy.percentile(score_differences_array, percentile)
        figure.add_hline(y=threshold)

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
    dcc.Slider(0, 100, value=85, id="percentile"),
    html.Div(id="data-slot"),
])

app.run()
