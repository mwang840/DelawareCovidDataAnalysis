import pandas as pd
import seaborn as sb
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

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

#Up here we have a good graph of the Covid cases throughout Delaware within the first four months the first case appeared in the state

#Now we will measure how often the cases increased throughout the state
sb.set_theme(style="darkgrid")
sb.lineplot(x=covidData["Delaware Day #"], y=covidData["DE infections"], data=covidData)
sb.lineplot(x=covidData["Delaware Day #"], y=covidData["DE increase in infections "], data=covidData, linewidth=1.5)
sb.lineplot(x=covidData["Delaware Day #"], y=covidData["DE Percent Increase"], data=covidData, linewidth=1.5)
#Now we will train a regression model given the data within Delaware
X = covidData.drop(columns=["Delaware Day #", "DE increase in infections ", "DE Percent Increase", "MD Day #","MD infections", "MD increase in infections","MD percent increase infections", "Califronia Day #", "CA infections", "CA increase in infections","CA percent increase","Texas Day #","Infections", "Increase in infections","Percent Increase"])
y = covidData["DE infections"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
svrScalar = StandardScaler()
X_train_scaled = svrScalar.fit_transform(X_train)
X_test_scaled = svrScalar.transform(X_test)
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)
finalAccuracyCovidCasesDe = r2_score(y_test.values, y_pred) * 100
print(f"The accuracy of training an SVR on the Delaware days and the covid cases throughout the state is {finalAccuracyCovidCasesDe}%")
finalAccuracyCovidCasesDe = mean_squared_error(y_test, y_pred) * 100
print(f"The accuracy of training an SVR on the Delaware days and the covid cases throughout the state is {finalAccuracyCovidCasesDe}%")