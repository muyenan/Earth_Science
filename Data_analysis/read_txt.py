from parameters import Parameters


class Read_txt:
    def __init__(self):
        parameters = Parameters()

        self.TXT_PATH = parameters.TXT_PATH
        self.TXT_ENCODING = parameters.ENCODING

    def read_txt(self, return_value):
        """return_value: 0 means return years and 1 means return annual anomaly."""
        years = []
        annual_anomaly = []
        txt_readable = False

        try:
            with open(self.TXT_PATH, "r", encoding=self.TXT_ENCODING) as f:
                lines = f.readlines()
                txt_readable = True
        except FileNotFoundError:
            print("File not found: " + self.TXT_PATH)
        except UnicodeDecodeError:
            print("File not readable with " + self.TXT_ENCODING)
        except Exception as e:
            print("Unknown Error: " + str(e))

        if txt_readable:
            for line in lines:
                parts = line.split()

                if len(parts) == 0:
                    continue

                if not parts[0].isdigit():
                    continue

                year = int(parts[0])
                jd_value = parts[13]

                if "*" in jd_value:
                    continue

                anomaly_c = float(jd_value) / 100

                years.append(year)
                annual_anomaly.append(anomaly_c)

        if return_value == 0:
            return years
        return annual_anomaly
