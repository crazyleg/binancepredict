import pytest
import torch
import vcr
import yaml
from api.binance import BinanceAPI
from helpers import prepare_data
from ml.models.BSM import BSM4

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


@vcr.use_cassette()
def test_inference():
    API = BinanceAPI(config=cfg["binance"])
    API.wait_for_1m_tick()

    data = API.get_inference_data(
        symbols=cfg["models"]["resnet"]["symbols"], interval="1m"
    )
    data = prepare_data(cfg, data)

    net = BSM4(
        n_features=data.shape[0], n_outputs=len(cfg["models"]["resnet"]["symbols"])
    )
    net.load_state_dict(torch.load("ml/models/saved_model.pth"))
    net.eval()
    data = torch.Tensor(data)[:, -cfg["models"]["resnet"]["window_size"] :].unsqueeze(0)
    with torch.no_grad():
        print(net(data))
