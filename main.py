import logging

import pandas as pd
import torch
import yaml
from logstash_async.handler import AsynchronousLogstashHandler
from prettylog import basic_config
from sklearn.linear_model import LinearRegression

from api.binance import BinanceAPI
from datastructures.tickerstream import TickerStream
from datastructures.ticketdata import TickerData
from datastructures.trading import Trading
from helpers import prepare_data
from ml.models.BSM import ResNetBSM4

# Configure logging
# TODO also log to file
basic_config(level=logging.DEBUG, buffered=False, log_format="color")
test_logger = logging.getLogger()
test_logger.setLevel(logging.INFO)
test_logger.addHandler(
    AsynchronousLogstashHandler("localhost", 5000, database_path="logstash.db")
)

test_logger.info("Starting prediction server")


with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# get data with API


# TODO blocker! delays have to be normalized
# TODO blocker! asses the losses of the 1000-512 stamps and log it!!!
# TODO log predictions and operations
# TODO log trading operations - currency, open_time, buy or sell
# TODO log profits - monitor 15 min, close operation and log profits
# TODO Dataclass for thresholds
# TODO


def run_prediction_loop():
    # LOG time - time to wait, getting the currency prices, log prices,
    # TODO check for maxtimestamp
    thresholds = pd.read_pickle("ml/models/thresholds.pkl")
    C_thresholds = pd.read_pickle("ml/models/C_thresholds.pkl")
    trading_engine = Trading(cfg, C_thresholds, test_logger)
    API = BinanceAPI(config=cfg["binance"])

    net = ResNetBSM4(
        n_features=len(cfg["models"]["resnet"]["symbols"]) * 9,  # TODO magic number
        n_outputs=len(cfg["models"]["resnet"]["symbols"]),
        features=32,
        blocks=3,
    )
    if torch.cuda.is_available():
        net.load_state_dict(torch.load("ml/models/BSM4.pth"))
    else:
        net.load_state_dict(
            torch.load("ml/models/BSM4.pth", map_location=torch.device("cpu"))
        )

    net.eval()

    while True:
        API.wait_for_1m_tick()

        data, max_timestamp = API.get_interence_data_async(
            symbols=cfg["models"]["resnet"]["symbols"], interval="1m"
        )

        data_for_inference, data_for_lr, returns = prepare_data(cfg, data)

        data_for_inference = torch.Tensor(data_for_inference).unsqueeze(0)

        with torch.no_grad():
            results = net(data_for_inference)
            results_for_lr = net(torch.Tensor(data_for_lr))

        # TODO as a features for LR I can also add last 15m returns (not future)
        # should be also good. But then maybe swapping LR to something non-linear should also help.
        # Also things like volume could be of use for LR.

        # TODO add a second trigger some conservative - threshold will be a quantile of historical predictions

        reg = LinearRegression().fit(results_for_lr, returns)
        results = reg.predict(results)

        trading_engine.update_status(max_timestamp, data, results)
        logging.info("cycle finished")


if __name__ == "__main__":
    run_prediction_loop()
