import joblib
import datetime
import os
import pandas as pd
#import yfinance as yf
from fbprophet import Prophet

TODAY = datetime.date.today()

def predict(ticker="MSFT", days=7):
    model_file = os.path.join('',(f"{ticker}.pkl"))
    #model_file = Path(BASE_DIR).joinpath(f"{ticker}.pkl")
    if not model_file == "MSFT.pkl":
        return False

    model_file = "MSFT.pkl"
    model = joblib.load(model_file)

    future = TODAY + datetime.timedelta(days=days)

    dates = pd.date_range(start="2020-01-01", end=future.strftime("%m/%d/%Y"),)
    df = pd.DataFrame({"ds": dates})

    forecast = model.predict(df)
    #model.plot(forecast).savefig(f"{ticker}_plot.png")
    #model.plot_components(forecast).savefig(f"{ticker}_plot_components.png")
    return forecast.tail(days).to_dict("records")


def convert(prediction_list):
    output = {}
    for data in prediction_list:
        date = data["ds"].strftime("%m/%d/%Y")
        output[date] = data["trend"]
    return output

def main():
    prediction_list = predict()
    print(convert(prediction_list))

if __name__ == "__main__":
    main()
