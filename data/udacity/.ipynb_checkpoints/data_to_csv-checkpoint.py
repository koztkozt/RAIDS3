# importiong the modules
import pandas as pd
import numpy as np
import math

df = pd.read_csv("./interpolated.csv", engine="python")
df = df[df["frame_id"] == "center_camera"]
df["filename"] = df["filename"].str[7:]
df["angle"] = pd.to_numeric(df["angle"])
df["angle_convert_org"] = df["angle"]
df["curve"] = np.where(abs(df['angle']) > 0.3, True, False)
df.to_csv("data.csv", index=False)
print(df.tail(n=10))
# print(df.describe())
