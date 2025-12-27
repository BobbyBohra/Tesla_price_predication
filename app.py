from flask import Flask, render_template, request
from model import train_and_predict
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    actual, predicted = train_and_predict()

    # Get user-selected number of points
    n_points = int(request.form.get('points', 100))
    actual = actual[-n_points:]
    predicted = predicted[-n_points:]

    # Dates for x-axis
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(actual)).strftime('%Y-%m-%d')

    # Plotly interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual, mode='lines+markers', name='Actual Price',
        line=dict(color='lime', width=3), marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=predicted, mode='lines+markers', name='Predicted Price',
        line=dict(color='red', width=3, dash='dash'), marker=dict(size=5)
    ))
    fig.update_layout(
        title='Tesla Stock Price Prediction (LSTM)',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=600,
        margin=dict(l=20, r=20, t=60, b=40),
        hovermode='x unified'
    )
    plot_div = pio.to_html(fig, full_html=False)

    # Color-coded table for last 20 points
    table_data = pd.DataFrame({'Actual': actual[-20:], 'Predicted': predicted[-20:]})
    def colorize(row):
        return ['background-color: lightgreen' if row['Actual'] > row['Predicted'] else 'background-color: salmon',
                'background-color: lightgreen' if row['Predicted'] > row['Actual'] else 'background-color: salmon']
    table_html = table_data.style.apply(colorize, axis=1).hide(axis="index")._repr_html_()

    # Stats cards
    stats = {
        'Last Actual': round(actual[-1], 2),
        'Last Predicted': round(predicted[-1], 2),
        'Difference': round(actual[-1]-predicted[-1], 2),
        'Average': round(sum(actual)/len(actual), 2)
    }

    return render_template('index.html', plot_div=plot_div, table_html=table_html, stats=stats, n_points=n_points)

if __name__ == '__main__':
    app.run(debug=True)
