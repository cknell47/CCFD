# Reading into data frame from file
import pandas as pd

data_frame = pd.read_csv('./creditcard.csv')

# Profiling dataset
from ydata_profiling import ProfileReport

profile = ProfileReport(data_frame, title="Profiling Report")
profile.to_file("./profiling/data_report.html")

