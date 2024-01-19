import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request

app = Flask(__name__)

def star_product(df):
    product_column = 'Product line'
    product_sales = df.groupby(product_column)['Total'].sum()
    star_product = product_sales.idxmax()

    return star_product

def demand_forecasting(df):
    features = df[['Unit price', 'Quantity', 'Tax 5%', 'gross income', 'Rating']]
    target = df['Total']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    df['Demand_Predictions'] = model.predict(features)
    order_recommendations = df.groupby(['City', 'Product line']).agg({
        'Demand_Predictions': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    return order_recommendations

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            try:
                df = pd.read_csv(file)
                star_product_result = star_product(df)
                order_recommendations_result = demand_forecasting(df)
                return render_template("result.html", star_product=star_product_result, order_recommendations=order_recommendations_result)
            except Exception as e:
                return render_template("upload.html", error=str(e))

    return render_template("upload.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080, debug=True)

