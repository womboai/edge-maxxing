import json

BASE_PORT = 8000

API_SERVICE_TEMPLATE = """
  api-{id}:
      restart: always
      
      build:
        context: ../
        dockerfile: ./validator/submission_tester/Dockerfile

      volumes:
        - ~/edge-maxxing/huggingface:/home/sandbox/.cache/huggingface

      ports:
        - 127.0.0.1:{port}:8000

      environment:
        VALIDATOR_HOTKEY_SS58_ADDRESS: $VALIDATOR_HOTKEY_SS58_ADDRESS
        VALIDATOR_DEBUG: $VALIDATOR_DEBUG
        CUDA_VISIBLE_DEVICES: 0

      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                device_ids: [ '{id}' ]
                capabilities: [ gpu ]
"""

VALIDATOR_SERVICE_TEMPLATE = """
  validator:
      restart: always
      
      build:
        context: ../
        dockerfile: ./validator/weight_setting/Dockerfile

      volumes:
        - ~/.bittensor:/home/validator/.bittensor
        - type: bind
          source: ~/.netrc
          target: /home/validator/.netrc

      command: --benchmarker_api {apis} $VALIDATOR_ARGS

      network_mode: host
"""


def main():
    with open("compose-gpu-layout.json") as f:
        layout: list[int] = json.load(f)

    with open("compose.yaml", "w") as compose:
        compose.write("services:")

        for device_id in layout:
            compose.write(API_SERVICE_TEMPLATE.format(id=device_id, port=str(BASE_PORT + device_id)))

        compose.write(
            VALIDATOR_SERVICE_TEMPLATE.format(
                apis=" ".join(
                    [
                        f"http://localhost:{BASE_PORT + device_id}"
                        for device_id in layout
                    ]
                )
            )
        )


if __name__ == '__main__':
    main()
