import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request

from . import Miner
from neuron import get_config, AVERAGE_TIME, CheckpointInfo

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
def get_checkpoint(request: Request) -> CheckpointInfo:
    return CheckpointInfo(
        repository=request.app.state.miner.checkpoint,
        average_time=AVERAGE_TIME,
    )


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
