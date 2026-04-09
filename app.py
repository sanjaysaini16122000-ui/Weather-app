import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web servers
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import io
import base64
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "secret_weather_key"

# Configuration
API_KEY = "fc2a440df17185166db00826c5f20b87"
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

def fetch_weather_data(city):
    params = {'q': city, 'appid': API_KEY, 'units': 'metric'}
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        weather_list = []
        for entry in data['list']:
            weather_list.append({
                'datetime': entry['dt_txt'],
                'temp': entry['main']['temp'],
                'humidity': entry['main']['humidity'],
                'pressure': entry['main']['pressure'],
                'rain': entry.get('rain', {}).get('3h', 0)  # Volume in mm
            })
        df = pd.DataFrame(weather_list)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def train_and_predict(df):
    X = np.array(df.index).reshape(-1, 1)
    y = df['temp'].values
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.array(range(len(df), len(df) + 10)).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    future_df = pd.DataFrame({
        'time_index': future_X.flatten(),
        'predicted_temp': predictions
    })
    return model, future_df

def create_plot(df, future_df, city):
    plt.figure(figsize=(10, 5), facecolor='none')
    plt.plot(df.index, df['temp'], label='Historical (3h steps)', color='#4f46e5', marker='o', linewidth=2)
    plt.plot(future_df['time_index'], future_df['predicted_temp'], label='Predicted Trend', color='#f43f5e', marker='x', linestyle='--', linewidth=2)
    
    plt.title(f"Temperature Prediction for {city.capitalize()}", color='white', fontsize=14, pad=20)
    plt.xlabel("Time Steps", color='#cbd5e1')
    plt.ylabel("Temp (°C)", color='#cbd5e1')
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
    plt.grid(True, linestyle='--', alpha=0.2)
    
    # Styling for dark mode web app
    ax = plt.gca()
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_humidity_plot(df, city):
    plt.figure(figsize=(10, 4), facecolor='none')
    plt.plot(df['datetime'], df['humidity'], color='#10b981', linewidth=2)
    plt.fill_between(df['datetime'], df['humidity'], color='#10b981', alpha=0.1)
    
    plt.title(f"Humidity Trend for {city.capitalize()}", color='white', fontsize=14, pad=15)
    plt.xlabel("Datetime", color='#cbd5e1')
    plt.ylabel("Humidity (%)", color='#cbd5e1')
    plt.xticks(rotation=45, color='#cbd5e1')
    plt.grid(True, linestyle='--', alpha=0.1)
    
    ax = plt.gca()
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_rain_plot(df, city):
    plt.figure(figsize=(10, 4), facecolor='none')
    plt.bar(df['datetime'], df['rain'], color='#38bdf8', alpha=0.8, width=0.08)
    plt.title(f"Rain Forecast for {city.capitalize()} (Next 5 Days)", color='white', fontsize=14, pad=15)
    plt.xlabel("Datetime", color='#cbd5e1')
    plt.ylabel("Rain Volume (mm)", color='#cbd5e1')
    plt.xticks(rotation=45, color='#cbd5e1')
    plt.grid(True, linestyle='--', alpha=0.1, axis='y')
    ax = plt.gca()
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city', 'Jaipur').strip()
        df = fetch_weather_data(city)
        
        if df is not None:
            model, future_df = train_and_predict(df)
            temp_plot = create_plot(df, future_df, city)
            humidity_plot = create_humidity_plot(df, city)
            rain_plot = create_rain_plot(df, city)
            
            weather_data = {
                'city': city,
                'current_temp': round(df['temp'].iloc[-1], 1),
                'current_humidity': df['humidity'].iloc[-1],
                'next_pred': round(future_df['predicted_temp'].iloc[0], 1),
                'temp_plot': temp_plot,
                'humidity_plot': humidity_plot,
                'rain_plot': rain_plot,
                'table_data': df.tail(10).to_dict('records') # Show more records for rain
            }
        else:
            flash(f"Could not find weather data for '{city}'. Please check the spelling.")

    return render_template('index.html', data=weather_data)

if __name__ == '__main__':
    app.run(debug=True)
