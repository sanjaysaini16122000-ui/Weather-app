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

        # 2. Fetch 5-DAY forecast (Restore this specifically for the TABLE)
        fore_resp = requests.get(BASE_URL, params=params)
        fore_resp.raise_for_status()
        forecast_data = fore_resp.json()
        
        # Prepare official table data (Original format)
        table_list = []
        for entry in forecast_data['list']:
            table_list.append({
                'datetime': pd.to_datetime(entry['dt_txt']),
                'temp': round(entry['main']['temp'], 1),
                'humidity': entry['main']['humidity'], 
                'rain': round(entry.get('rain', {}).get('3h', 0), 1)
            })
        
        # 3. Fetch HISTORICAL + FORECAST data from Open-Meteo (1 day past + 7 days forecast)
        lat, lon = current_data['coord']['lat'], current_data['coord']['lon']
        # We only fetch 1 past day (April 9th onwards)
        meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation&past_days=1"
        meteo_resp = requests.get(meteo_url)
        meteo_data = meteo_resp.json()
        
        # Process Open-Meteo data into a high-resolution DataFrame
        hourly = meteo_data['hourly']
        full_df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'temp': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'rain': hourly['precipitation']
        })
        
        # Show 5 Days Total (Yesterday + Today + 3 Future) for maximum detail
        start_date = full_df['datetime'].min()
        end_date = start_date + pd.Timedelta(days=5)
        df = full_df[full_df['datetime'] < end_date].copy()
        
        return df, current_data, table_list
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
    plt.figure(figsize=(14, 7), facecolor='none') # Taller and Wider
    
    # Identify index of "Now"
    now = pd.Timestamp.now()
    df = df.sort_values('datetime')
    
    # Plot historical data (Past)
    past_df = df[df['datetime'] <= now]
    future_api_df = df[df['datetime'] > now]
    
    plt.plot(past_df['datetime'], past_df['temp'], label='Observed (Past)', color='#6366f1', linewidth=2.5)
    plt.fill_between(past_df['datetime'], past_df['temp'], color='#6366f1', alpha=0.1)
    
    # Plot Official Forecast
    plt.plot(future_api_df['datetime'], future_api_df['temp'], label='Forecast (Official)', color='#4f46e5', marker='o', markersize=3, linewidth=2.5)
    
    # Plot AI Trend prediction
    plt.plot(future_df['datetime'], future_df['predicted_temp'], label='AI Extension', color='#f43f5e', linestyle='--', linewidth=2.5)
    
    # Add a "NOW" vertical line
    plt.axvline(x=now, color='#fbbf24', linestyle='--', label=f'PRESENT', linewidth=2.5, zorder=5)
    
    plt.title(f"Temperature Journey for {city.capitalize()}", color='white', fontsize=28, pad=45, fontweight='bold')
    plt.xlabel("Timeline", color='#cbd5e1', labelpad=20, fontsize=20)
    plt.ylabel("Temp (°C)", color='#cbd5e1', fontsize=20)
    
    # Enhanced Grid
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.grid(True, which='major', linestyle='-', alpha=0.15)
    plt.grid(True, which='minor', linestyle=':', alpha=0.05)
    
    plt.xticks(rotation=0, horizontalalignment='center', fontsize=16) # Ultra Large Ticks
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', loc='upper left', fontsize=16)
    
    # Dark mode spines
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1', labelsize=16) # Ultra Large Numbers
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_humidity_plot(df, city):
    plt.figure(figsize=(14, 6.5), facecolor='none') # Taller
    df = df.sort_values('datetime')
    now = pd.Timestamp.now()
    
    plt.plot(df['datetime'], df['humidity'], color='#10b981', linewidth=2.5, label='Humidity %')
    plt.fill_between(df['datetime'], df['humidity'], color='#10b981', alpha=0.1)
    plt.axvline(x=now, color='#fbbf24', linestyle='--', label='PRESENT', linewidth=2.5, zorder=5)
    
    plt.title(f"Humidity Journey for {city.capitalize()}", color='white', fontsize=26, fontweight='bold', pad=35)
    plt.ylabel("Humidity (%)", color='#cbd5e1', fontsize=18)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.grid(True, which='major', linestyle='-', alpha=0.1)
    plt.grid(True, which='minor', linestyle=':', alpha=0.05)
    plt.xticks(rotation=0, fontsize=16)
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=16)
    
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1', labelsize=16)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_rain_plot(df, city):
    plt.figure(figsize=(14, 6.5), facecolor='none') # Taller
    df = df.sort_values('datetime')
    now = pd.Timestamp.now()
    
    plt.bar(df['datetime'], df['rain'], color='#38bdf8', alpha=0.8, width=0.08, label='Rain (mm)')
    plt.axvline(x=now, color='#fbbf24', linestyle='--', label='PRESENT', linewidth=2.5, zorder=5)
    
    plt.title(f"Precipitation Journey for {city.capitalize()}", color='white', fontsize=26, fontweight='bold', pad=35)
    plt.ylabel("Rain (mm)", color='#cbd5e1', fontsize=18)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.grid(True, alpha=0.1, axis='y')
    plt.xticks(rotation=0, fontsize=16)
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=16)
    
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#cbd5e1', labelsize=16)
    
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
        city = request.form.get('city')
        df, current_api_data, table_list = fetch_weather_data(city)
        
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
                'next_pred': round(future_df['predicted_temp'].iloc[1], 1), # Use first actual prediction
                'temp_plot': temp_plot,
                'humidity_plot': humidity_plot,
                'rain_plot': rain_plot,
                'ai_advice': ai_advice,
                'lat': current_api_data['coord']['lat'],
                'lon': current_api_data['coord']['lon'],
                'weather_desc': current_api_data['weather'][0]['description'].capitalize(),
                'weather_icon': current_api_data['weather'][0]['icon'],
                'current_rain': current_api_data.get('rain', {}).get('1h', 0),
                'api_key': API_KEY,
                'table_data': table_list # Use the official OWM table list
            }
        else:
            flash(f"Could not find weather data for '{city}'. Please check the spelling.")

    return render_template('index.html', data=weather_data)

if __name__ == '__main__':
    app.run(debug=True)
