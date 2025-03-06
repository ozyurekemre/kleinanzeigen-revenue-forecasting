##################################################################
# Kleinanzeigen Case Study - 2023 Revenue Overview/ 2024 Forecast
#################################################################

##################################
# DISCOVERY DATA ANALYSIS
##################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

accounts_df = pd.read_csv("datasets/cart/Accounts.csv", delimiter=",")
contracts_df = pd.read_csv("datasets/cart/Contracts.csv", delimiter=",")
services_df = pd.read_csv("datasets/cart/Services.csv", delimiter=",", encoding="ISO-8859-9")
invoices_df = pd.read_csv("datasets/cart/Invoices.csv", delimiter=",", encoding="ISO-8859-9")
accounts_df.head()
contracts_df.head()
services_df.head()
invoices_df.head()

services_df.columns = services_df.columns.str.replace('ï»¿', '')
invoices_df.columns = invoices_df.columns.str.replace('ï»¿', '')


invoices_df['Invoice Revenue'] = invoices_df['Invoice Revenue'].replace({r'\s+':'.'}, regex=True)
invoices_df['Invoice Revenue'] = pd.to_numeric(invoices_df['Invoice Revenue'], errors='coerce')
invoices_df['Invoice Revenue'] = invoices_df['Invoice Revenue'].astype(float)

services_df['Service Start Date'] = pd.to_datetime(services_df['Service Start Date'], errors='coerce')
services_df['Service End Date'] = pd.to_datetime(services_df['Service End Date'], errors='coerce')
invoices_df['Invoice Period'] = pd.to_datetime(invoices_df['Invoice Period'], errors='coerce')

#Joins
df_1 = pd.merge(services_df,invoices_df, how='left', left_on='Service ID', right_on='Service ID')
df_2 = pd.merge(contracts_df, df_1, how='left', left_on='Contract ID', right_on='Contract ID')
df_final = pd.merge(accounts_df, df_2, how='left', left_on='Account ID', right_on='Account ID')

df_final.head()

##################################
# GENERAL OVERVIEW
##################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df_final)

##################################
# CAPTURE OF NUMERICAL AND CATEGORY VARIABLES
##################################
def grab_col_names(dataframe, cat_th=7, car_th=10):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df_final)

##################################
# ANALYSIS OF CATEGORY VARIABLES
##################################


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df_final, col)

##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_cols = ['Invoice Revenue']
for col in num_cols:
    num_summary(df_final, col)

##################################
# KPI CALCULATIONS
##################################
# Total Revenue
invoices_2023 = invoices_df[invoices_df['Invoice Period'].notnull() &
                            (invoices_df['Invoice Period'].dt.year == 2023)]
total_revenue = invoices_2023['Invoice Revenue'].sum()
print("Total Revenue (2023): €{:,.2f}".format(total_revenue))

#Active Account
df_final['Unique Customer ID'] = df_final['Account Parent ID'].fillna(df_final['Account ID'])

df_final['active_invoice'] = df_final['Invoice Period'].notnull() & (df_final['Invoice Period'].dt.year == 2023)

account_status = df_final.groupby('Unique Customer ID').agg(
    any_active_invoice=('active_invoice', 'any')
)
account_status['active'] = account_status['any_active_invoice']

active_customers = account_status['active'].sum()
print("Active Customers (2023, invoice-based): {:,}".format(active_customers))

# Contract Renewal Rate
contract_counts = contracts_df.groupby('Account ID')['Contract ID'].count()
renewal_rate = (contract_counts[contract_counts > 1].count() / contract_counts.count()) * 100

# Average Invoice Amount
average_invoice_amount = invoices_df['Invoice Revenue'].mean()


# Churn Rate
churn_rate = (services_df['Service Termination Reason'].notna().sum() / services_df.shape[0]) * 100

# ARPA (Average Revenue Per Active Account)
arpa = total_revenue / active_customers if active_customers > 0 else 0
print("ARPA (2023): €{:,.2f}".format(arpa))

# Service Product Type Disturibition
revenue_by_product = df_final.groupby('Service Product Type')['Invoice Revenue'].sum().reset_index()
print("Revenue by Service Product Type:")
print(revenue_by_product)


revenue_by_product = unique_invoices.groupby('Service Product Type')['Invoice Revenue'].sum().reset_index()
print(revenue_by_product)

new_contracts = contracts_df[contracts_df['Contract Type'] == 'New Customer']['Contract ID'].nunique()
existing_contracts = contracts_df[contracts_df['Contract Type'] == 'Existing Customer']['Contract ID'].nunique()
print("New Customer Contracts:", new_contracts)
print("Existing Customer Contracts:", existing_contracts)

# Some Kpi results
kpi_results = {
    "Total Revenue": total_revenue,
    "Active Customers": active_customers,
    "Contract Renewal Rate (%)": renewal_rate,
    "Average Invoice Amount": average_invoice_amount,
    "Churn Rate (%)": churn_rate,
}
print("KPI Results:")
for key, value in kpi_results.items():
    print(f"{key}: {value}")

##################################
# MODELLING
##################################
invoices_2023 = invoices_df[
    invoices_df['Invoice Period'].notnull() &
    (invoices_df['Invoice Period'].dt.year == 2023)
].copy()

invoices_2023['Month'] = invoices_2023['Invoice Period'].dt.to_period('M').dt.to_timestamp()

monthly_revenue = invoices_2023.groupby('Month')['Invoice Revenue'].sum().reset_index()

print(monthly_revenue)

monthly_revenue_1= monthly_revenue.set_index('Month')['Invoice Revenue']
monthly_revenue_1 = monthly_revenue_1.asfreq('MS') #monthly frekans

# First option - mean()
avg_value = monthly_revenue_1.loc['2023-01-01':'2023-11-01'].mean()
monthly_revenue_1.loc['2023-12-01'] = avg_value    # Dec - mean()

model = ARIMA(monthly_revenue_1, order=(1,1,1))
results = model.fit()

forecast_steps = 12
arima_forecast = results.forecast(steps=forecast_steps)
print(arima_forecast)


# Second option -Sarimax - dummy

monthly_df = monthly_revenue_1.reset_index().copy()
monthly_df['dec2023_outlier'] = 0
monthly_df.loc[monthly_df['Month'] == '2023-12-01','dec2023_outlier'] = 1

monthly_df.set_index('Month', inplace=True)
endog = monthly_df['Invoice Revenue'].asfreq('MS')
exog = monthly_df[['dec2023_outlier']].asfreq('MS')


model = SARIMAX(endog,
                exog=exog,
                order=(1,1,1),
                seasonal_order=(1,1,1,12),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()


future_dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')
future_exog = pd.DataFrame({'dec2023_outlier': [0]*12}, index=future_dates)

forecast = results.forecast(steps=12, exog=future_exog)

print(forecast)

#Non-seasonal Sarima
model = SARIMAX(endog,
                exog=exog,
                order=(1,1,1),
                seasonal_order=(0,0,0,0),  # MEVSİMSELLİK YOK
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()

future_dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')
future_exog = pd.DataFrame({'dec2023_outlier': [0]*12}, index=future_dates)

forecast = results.forecast(steps=12, exog=future_exog)
print(forecast)





























































