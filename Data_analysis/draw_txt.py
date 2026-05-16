import os

import matplotlib.pyplot as plt
import pandas as pd

from parameters import Parameters


class Draw_txt:
    def __init__(self):
        parameters = Parameters()

        self.RESULT_PATH = parameters.RESULT_PATH

        if not os.path.exists(self.RESULT_PATH):
            os.makedirs(self.RESULT_PATH)

    def draw_txt(self, years, annual_anomaly):
        try:
            average = pd.Series(annual_anomaly).rolling(window=5, center=True).mean()

            plt.figure(figsize=(20, 10))

            plt.plot(years, annual_anomaly, color="blue", label="annual anomaly")
            plt.plot(years, average, color="red", label="5-year moving average")

            plt.axhline(y=0, color="gray", linestyle="--")

            plt.title("Global Annual Mean Temperature Anomaly Since 1880")
            plt.xlabel("Year")
            plt.ylabel("Temperature anomaly (deg C)")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(
                self.RESULT_PATH,
                "Global Annual Mean Temperature Anomaly.png"
            )
            plt.savefig(output_path, dpi=300)
            plt.close()

            print("Plot saved at " + output_path)
            return True
        except Exception as e:
            print("Encounter mistakes when plotting: " + str(e))
            return False
