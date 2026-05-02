import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import numpy as np
import io
import base64
from flask import Flask, render_template, request, flash
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secret_weather_key"

# Configuration
API_KEY = "fc2a440df17185166db00826c5f20b87"
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

def fetch_weather_data(city):
    params = {'q': city, 'appid': API_KEY, 'units': 'metric'}
    try:
        # 1. Fetch CURRENT weather for the main dashboard box
        curr_url = "https://api.openweathermap.org/data/2.5/weather"
        curr_resp = requests.get(curr_url, params=params, timeout=10)
        curr_resp.raise_for_status()
        current_data = curr_resp.json()

        # 2. Fetch 5-DAY forecast
        fore_resp = requests.get(BASE_URL, params=params, timeout=10)
        fore_resp.raise_for_status()
        forecast_data = fore_resp.json()
        
        # Prepare official table data
        table_list = []
        for entry in forecast_data['list']:
            table_list.append({
                'datetime': pd.to_datetime(entry['dt_txt']),
                'temp': round(entry['main']['temp'], 1),
                'humidity': entry['main']['humidity'], 
                'rain': round(entry.get('rain', {}).get('3h', 0), 1),
                'wind': round(entry['wind']['speed'], 1),
                'desc': entry['weather'][0]['description'].capitalize(),
                'icon': entry['weather'][0]['icon']
            })
        
        # 3. Fetch HISTORICAL + FORECAST data from Open-Meteo
        lat, lon = current_data['coord']['lat'], current_data['coord']['lon']
        meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation&past_days=1"
        meteo_resp = requests.get(meteo_url, timeout=10)
        meteo_resp.raise_for_status()
        meteo_data = meteo_resp.json()
        
        hourly = meteo_data['hourly']
        full_df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'temp': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'rain': hourly['precipitation']
        })
        
        # Show 5 Days Total
        start_date = full_df['datetime'].min()
        end_date = start_date + pd.Timedelta(days=5)
        df = full_df[full_df['datetime'] < end_date].copy()
        
        return df, current_data, table_list
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return "not_found", None, None
        return None, None, None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None

def train_and_predict(df):
    # Ensure index is numeric for regression
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['temp'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 24 hours (8 points of 3-hour intervals)
    future_steps = 12
    future_X = np.array(range(len(df), len(df) + future_steps)).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    last_dt = df['datetime'].iloc[-1]
    future_dates = [last_dt + pd.Timedelta(hours=3 * i) for i in range(1, future_steps + 1)]
    
    # Include last point for visual continuity
    future_dates.insert(0, last_dt)
    predictions = np.insert(predictions, 0, df['temp'].iloc[-1])
    
    future_df = pd.DataFrame({
        'datetime': future_dates,
        'predicted_temp': predictions
    })
    return model, future_df

def create_plot(df, future_df, city):
    plt.figure(figsize=(14, 5.5), facecolor='none')
    now = pd.Timestamp.now()
    df = df.sort_values('datetime')
    
    past_df = df[df['datetime'] <= now]
    future_api_df = df[df['datetime'] > now]
    
    plt.plot(past_df['datetime'], past_df['temp'], label='Historical', color='#818cf8', linewidth=3, alpha=0.8)
    plt.fill_between(past_df['datetime'], past_df['temp'], color='#818cf8', alpha=0.1)
    
    plt.plot(future_api_df['datetime'], future_api_df['temp'], label='Forecast', color='#4f46e5', marker='o', markersize=4, linewidth=3)
    plt.plot(future_df['datetime'], future_df['predicted_temp'], label='AI Trend', color='#f43f5e', linestyle='--', linewidth=3)
    
    plt.axvline(x=now, color='#fbbf24', linestyle='-', label='NOW', linewidth=2, zorder=5)
    
    plt.title(f"Temperature Trajectory: {city.capitalize()}", color='white', fontsize=22, pad=30, fontweight='bold')
    plt.ylabel("Temp (°C)", color='#94a3b8', fontsize=14)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a, %b %d'))
    
    plt.grid(True, which='major', linestyle='-', alpha=0.1)
    plt.xticks(color='#94a3b8', fontsize=12)
    plt.yticks(color='#94a3b8', fontsize=12)
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', loc='upper left', framealpha=0.8)
    
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True, dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_humidity_plot(df, city):
    plt.figure(figsize=(14, 5.5), facecolor='none')
    df = df.sort_values('datetime')
    
    plt.plot(df['datetime'], df['humidity'], color='#10b981', linewidth=3, label='Humidity %')
    plt.fill_between(df['datetime'], df['humidity'], color='#10b981', alpha=0.1)
    
    plt.title(f"Humidity Levels: {city.capitalize()}", color='white', fontsize=20, fontweight='bold', pad=25)
    plt.ylabel("%", color='#94a3b8', fontsize=14)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a, %b %d'))
    
    plt.grid(True, alpha=0.1)
    plt.xticks(color='#94a3b8', fontsize=12)
    plt.yticks(color='#94a3b8', fontsize=12)
    
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True, dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_rain_plot(df, city):
    plt.figure(figsize=(14, 5.5), facecolor='none')
    df = df.sort_values('datetime')
    
    plt.bar(df['datetime'], df['rain'], color='#38bdf8', alpha=0.7, width=0.08, label='Rain (mm)')
    
    plt.title(f"Precipitation Forecast: {city.capitalize()}", color='white', fontsize=20, fontweight='bold', pad=25)
    plt.ylabel("mm", color='#94a3b8', fontsize=14)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a, %b %d'))
    
    plt.grid(True, alpha=0.1, axis='y')
    plt.xticks(color='#94a3b8', fontsize=12)
    plt.yticks(color='#94a3b8', fontsize=12)
    
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True, dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def generate_ai_advisory(current_data, future_df, city):
    temp = current_data['main']['temp']
    humidity = current_data['main']['humidity']
    weather_main = current_data['weather'][0]['main'].lower()
    
    # ML Insight
    next_3h_pred = future_df['predicted_temp'].iloc[1] if len(future_df) > 1 else temp
    trend = "rising" if next_3h_pred > temp + 0.5 else "falling" if next_3h_pred < temp - 0.5 else "stable"
    
    advisories = []
    
    if "rain" in weather_main or "drizzle" in weather_main:
        advisories.append("Precipitation detected. An umbrella is essential today.")
    elif "clear" in weather_main and temp > 30:
        advisories.append("High UV levels expected. Wear sunscreen and stay hydrated.")
    
    if temp > 38:
        advisories.append("Extreme heat warning! Minimize outdoor activities.")
    elif temp < 10:
        advisories.append("Cold conditions. Layer up to stay warm.")
        
    if humidity > 80:
        advisories.append("High humidity may cause discomfort. Opt for breathable clothing.")
        
    if trend != "stable":
        diff = abs(next_3h_pred - temp)
        advisories.append(f"AI models predict a {trend} temperature trend ({round(diff, 1)}°C shift soon).")
    else:
        advisories.append("Temperature stability is expected for the next few hours.")
        
    return " ".join(advisories)

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city')
        result = fetch_weather_data(city)
        
        if result == "not_found":
            flash(f"Location '{city}' not found. Please try another city.")
        elif result[0] is not None:
            df, current_api_data, table_list = result
            model, future_df = train_and_predict(df)
            
            weather_data = {
                'city': city,
                'current_temp': round(current_api_data['main']['temp'], 1),
                'feels_like': round(current_api_data['main']['feels_like'], 1),
                'current_humidity': current_api_data['main']['humidity'],
                'current_wind': round(current_api_data['wind']['speed'], 1),
                'current_pressure': current_api_data['main']['pressure'],
                'visibility': round(current_api_data.get('visibility', 0) / 1000, 1),
                'next_pred': round(future_df['predicted_temp'].iloc[1], 1),
                'temp_plot': create_plot(df, future_df, city),
                'humidity_plot': create_humidity_plot(df, city),
                'rain_plot': create_rain_plot(df, city),
                'ai_advice': generate_ai_advisory(current_api_data, future_df, city),
                'lat': current_api_data['coord']['lat'],
                'lon': current_api_data['coord']['lon'],
                'weather_desc': current_api_data['weather'][0]['description'].capitalize(),
                'weather_icon': current_api_data['weather'][0]['icon'],
                'current_rain': current_api_data.get('rain', {}).get('1h', 0),
                'api_key': API_KEY,
                'table_data': table_list,
                'unique_dates': sorted(list(set(item['datetime'].strftime('%d %b') for item in table_list)), 
                                     key=lambda x: datetime.strptime(x + f" {datetime.now().year}", '%d %b %Y'))
            }
        else:
            flash("Service temporarily unavailable. Please try again later.")

    return render_template('index.html', data=weather_data)

if __name__ == '__main__':
    app.run(debug=True)
