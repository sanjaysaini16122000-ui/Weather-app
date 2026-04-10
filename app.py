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
        # 1. Fetch CURRENT weather for the main dashboard box
        curr_url = "http://api.openweathermap.org/data/2.5/weather"
        curr_resp = requests.get(curr_url, params=params)
        curr_resp.raise_for_status()
        current_data = curr_resp.json()

        # 2. Fetch 5-DAY forecast for current display stats
        fore_resp = requests.get(BASE_URL, params=params)
        fore_resp.raise_for_status()
        forecast_data = fore_resp.json()
        
        # 3. Fetch HISTORICAL + FORECAST data from Open-Meteo (3 days past + 7 days future)
        lat, lon = current_data['coord']['lat'], current_data['coord']['lon']
        meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation&past_days=4"
        meteo_resp = requests.get(meteo_url)
        meteo_data = meteo_resp.json()
        
        # Process Open-Meteo data into a high-resolution DataFrame
        hourly = meteo_data['hourly']
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'temp': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'rain': hourly['precipitation'] # Added this to fix KeyError
        })
        
        return df, current_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

def train_and_predict(df):
    X = np.array(df.index).reshape(-1, 1)
    y = df['temp'].values
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.array(range(len(df), len(df) + 11)).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Generate future datetimes starting from the last point + 3 hours
    last_dt = df['datetime'].iloc[-1]
    future_dates = [last_dt + pd.Timedelta(hours=3 * i) for i in range(1, 12)]
    
    # Include the current last point in prediction to connect the lines seamlessly
    future_dates.insert(0, last_dt)
    predictions = np.insert(predictions, 0, df['temp'].iloc[-1])
    
    future_df = pd.DataFrame({
        'datetime': future_dates,
        'predicted_temp': predictions
    })
    return model, future_df

def create_plot(df, future_df, city):
    plt.figure(figsize=(10, 5), facecolor='none')
    
    # Identify index of "Now" (closest to current time)
    now = pd.Timestamp.now()
    # Ensure df is sorted by time
    df = df.sort_values('datetime')
    
    # Plot historical data (Past - Shaded differently)
    past_df = df[df['datetime'] <= now]
    future_api_df = df[df['datetime'] > now]
    
    plt.plot(past_df['datetime'], past_df['temp'], label='Observed (Past 3-4 Days)', color='#6366f1', linewidth=2)
    plt.fill_between(past_df['datetime'], past_df['temp'], color='#6366f1', alpha=0.1)
    
    # Plot Official Forecast
    plt.plot(future_api_df['datetime'], future_api_df['temp'], label='Official Forecast (Next 7 Days)', color='#4f46e5', marker='o', markersize=3, linewidth=2)
    
    # Plot AI Trend prediction
    plt.plot(future_df['datetime'], future_df['predicted_temp'], label='AI Linear Trend (Extension)', color='#f43f5e', linestyle='--', linewidth=2)
    
    # Add a "NOW" vertical line precisely at current time
    plt.axvline(x=now, color='#fbbf24', linestyle='--', label=f'PRESENT: {now.strftime("%b %d")}', linewidth=2, zorder=5)
    
    plt.title(f"Temperature Journey: Past, Present & Future for {city.capitalize()}", color='white', fontsize=14, pad=20, fontweight='bold')
    plt.xlabel("Timeline", color='#cbd5e1')
    plt.ylabel("Temp (°C)", color='#cbd5e1')
    
    # Fix the Legend
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.1)
    
    # Styling for dark mode
    ax = plt.gca()
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1', labelsize=8)
    
    # Format X-axis to show readable dates
    plt.xticks(rotation=45)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_humidity_plot(df, city):
    plt.figure(figsize=(10, 4), facecolor='none')
    
    # Sort and split data for coloring
    df = df.sort_values('datetime')
    now = pd.Timestamp.now()
    
    plt.plot(df['datetime'], df['humidity'], color='#10b981', linewidth=2, label='Humidity %')
    plt.fill_between(df['datetime'], df['humidity'], color='#10b981', alpha=0.1)
    
    # Add a "NOW" vertical line
    plt.axvline(x=now, color='#fbbf24', linestyle='--', label=f'PRESENT: {now.strftime("%b %d")}', linewidth=2, zorder=5)
    
    plt.title(f"Humidity Levels for {city.capitalize()}", color='white', fontsize=14, pad=15)
    plt.xlabel("Datetime", color='#cbd5e1')
    plt.ylabel("Humidity (%)", color='#cbd5e1')
    plt.xticks(rotation=45, color='#cbd5e1')
    plt.grid(True, linestyle='--', alpha=0.1)
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
    
    ax = plt.gca()
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1', labelsize=8)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_rain_plot(df, city):
    plt.figure(figsize=(10, 4), facecolor='none')
    
    # Sort data
    df = df.sort_values('datetime')
    now = pd.Timestamp.now()
    
    # Use bar chart for rain volume
    plt.bar(df['datetime'], df['rain'], color='#38bdf8', alpha=0.8, width=0.08, label='Rain Volume (mm)')
    
    # Add a "NOW" vertical line
    plt.axvline(x=now, color='#fbbf24', linestyle='--', label=f'PRESENT: {now.strftime("%b %d")}', linewidth=2, zorder=5)
    
    plt.title(f"Rain Volume Forecast for {city.capitalize()}", color='white', fontsize=14, pad=15)
    plt.xlabel("Datetime", color='#cbd5e1')
    plt.ylabel("Rain Volume (mm)", color='#cbd5e1')
    plt.xticks(rotation=45, color='#cbd5e1')
    plt.grid(True, linestyle='--', alpha=0.1, axis='y')
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
    
    ax = plt.gca()
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1', labelsize=8)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def generate_ai_advisory(current_data, future_df, city):
    temp = current_data['main']['temp']
    humidity = current_data['main']['humidity']
    rain = current_data.get('rain', {}).get('1h', 0)
    
    # ML Insight
    next_3h_pred = future_df['predicted_temp'].iloc[0]
    trend = "rising" if next_3h_pred > temp else "falling"
    diff = abs(next_3h_pred - temp)
    
    advice = f"In {city.capitalize()}, it's currently {round(temp)}°C. "
    
    if temp > 35:
        advice += "The heat is intense—stay hydrated and avoid direct sun. "
    elif temp < 15:
        advice += "It's quite chilly—keep yourself warm. "
    else:
        advice += "The temperature is quite comfortable for outdoor activities. "
        
    if humidity > 70:
        advice += "High humidity might make it feel a bit muggy. "
    
    if rain > 0:
        advice += "There's some rain in the area, don't forget an umbrella! "
        
    if diff > 1.5:
        advice += f"AI analysis shows a {trend} trend, expect it to get {trend} by ~{round(diff)}°C soon."
    else:
        advice += "The temperature seems stable for the next few hours."
        
    return advice

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city', 'Jaipur').strip()
        df, current_api_data = fetch_weather_data(city)
        
        if df is not None:
            model, future_df = train_and_predict(df)
            temp_plot = create_plot(df, future_df, city)
            humidity_plot = create_humidity_plot(df, city)
            rain_plot = create_rain_plot(df, city)
            ai_advice = generate_ai_advisory(current_api_data, future_df, city)
            
            weather_data = {
                'city': city,
                'current_temp': round(current_api_data['main']['temp'], 1),
                'current_humidity': current_api_data['main']['humidity'],
                'next_pred': round(future_df['predicted_temp'].iloc[0], 1),
                'temp_plot': temp_plot,
                'humidity_plot': humidity_plot,
                'rain_plot': rain_plot,
                'ai_advice': ai_advice,
                'lat': current_api_data['coord']['lat'],
                'lon': current_api_data['coord']['lon'],
                'weather_desc': current_api_data['weather'][0]['description'].capitalize(),
                'weather_icon': current_api_data['weather'][0]['icon'],
                'current_rain': current_api_data.get('rain', {}).get('1h', 0),
                'api_key': API_KEY, # Needed for browser-side tile loading
                'table_data': df.head(15).to_dict('records') 
            }
        else:
            flash(f"Could not find weather data for '{city}'. Please check the spelling.")

    return render_template('index.html', data=weather_data)

if __name__ == '__main__':
    app.run(debug=True)
