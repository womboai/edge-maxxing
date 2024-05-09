from os.path import basename

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.responses import Response
from requests_toolbelt import MultipartEncoder

from . import Miner
from neuron import get_config, AVERAGE_TIME

app = FastAPI()
scheduler = AsyncIOScheduler()


@app.on_event("startup")
def startup():
    miner = Miner(app.state.config)

    del app.state.config

    app.state.miner = miner

    scheduler.add_job(
        miner.sync,
        trigger="cron",
        second=f"*/{miner.config.neuron.epoch_length * 12}",
    )

    scheduler.start()


@app.get("checkpoint")
def get_checkpoint(request: Request) -> Response:
    response = MultipartEncoder(
        fields={
            "average_speed": (None, str(AVERAGE_TIME), "application/json"),

            "checkpoint": (
                basename(request.app.state.miner.checkpoint_path),
                open(request.app.state.miner.checkpoint_path, "rb"),
                "application/octet-stream"
            ),
        },
    )

    return Response(response.to_string(), media_type=response.content_type)


def main():
    config = get_config(Miner)

    app.state.config = config

    # Convert to IPv4 if the default, which happens to be IPv6
    ip = "0.0.0.0" if config.axon.ip == "[::]" else config.axon.ip

    uvicorn.run(
        app,
        host=ip,
        port=config.axon.port,
    )


if __name__ == '__main__':
    main()
