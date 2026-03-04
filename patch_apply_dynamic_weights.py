import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    old_weights = """                # Weights: NWS and HRRR are ground truth / high-res models for US, give them 2x weight
                SOURCE_WEIGHTS = {
                    "nws": 2.0,
                    "nws_hourly": 2.0,
                    "hrrr": 2.0,
                    "open_meteo": 1.0,
                    "tomorrow_io": 1.0,
                    "visual_crossing": 1.0,
                    "weatherbit": 1.0,
                    "openweathermap": 1.0
                }"""
                
    new_weights = """                # Get dynamic weights based on historical accuracy
                SOURCE_WEIGHTS = self.get_dynamic_weights(city_key)"""
                
    if old_weights in content:
        content = content.replace(old_weights, new_weights)
        with open(filepath, 'w') as f:
            f.write(content)
        print("Successfully applied dimport sys

def patch_file(filepath):
    wiai
def patcply    with open(filepath, ch        content = f.read()

    o/C
    old_weights = """   ps Betting/backend/app/services/weather_data.py')
