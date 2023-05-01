from myModule.BitcoinPredictionModel import BTCValuePredictor
from tensorboard import notebook


class TestBTCPricePredictor:
    btc_predictor = BTCValuePredictor('BTC-USD (4).csv')

    btc_predictor.prepare_data()

    btc_predictor.create_model(learning_rate=0.01)

    btc_predictor.train_model(batch_size=32, epochs=100)

    btc_predictor.predict()

    btc_predictor.plot_predictions()

    btc_predictor.plot_loss()

    notebook.list()

    # Start TensorBoard
    notebook.start("--logdir logs")
