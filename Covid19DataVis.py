import pandas as pd
import seaborn as sb
import numpy as np
#Load in my custom covid dataset
covidData = pd.read_csv("Covid 19 Infection rate .xlsx - Sheet1.csv")
#Print out the first five columns of the infections from delaware side
print(covidData["DE infections"].head(5))

#Find the mean of the Delaware Infections
covidDEMean = covidData["DE infections"].mean().round(0)
print(f"Mean of Covid 19 infections through Delaware during the first months of covid is {covidDEMean}")
#Find the median of the Delaware Infections
covidDEMedian = covidData["DE infections"].median().round(0)
print(f"Median of Covid 19 infections through Delaware during the first months of covid is {covidDEMedian}")

#Set a line graph for Delaware Covid Data
sb.set_theme(style="darkgrid")
sb.lineplot(x=covidData["Delaware Day #"], y=covidData["DE infections"], data=covidData)
