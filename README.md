# AI-Based Weather Forecasting System

A Python-based system that fetches real-time weather forecasts from OpenWeatherMap and uses Scikit-Learn's Linear Regression to predict future temperature trends.

## Features
- **Real-time Data**: Fetches 5-day / 3-hour forecast data.
- **Machine Learning**: Implements Linear Regression to model temperature changes over time.
- **Data Processing**: Uses Pandas for structured data manipulation.
- **Visualization**: Generates professional graphs for temperature trends and humidity.
- **Modular Design**: Clean, function-based architecture for easy extension.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Visit `http://127.0.0.1:5000` in your web browser.

## Project Structure
- `app.py`: Flask server with ML logic.
- `templates/index.html`: Modern Glassmorphism UI.
- `requirements.txt`: List of Python libraries.

## Technologies Used
- **Requests**: For API interaction.
- **Pandas**: For data structures.
- **Scikit-Learn**: For the Linear Regression model.
- **Matplotlib**: For plotting data.
