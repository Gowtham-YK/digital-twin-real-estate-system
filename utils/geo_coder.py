import requests

def geocode_location(location_name):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        
        params = {
            "q": location_name + ", Bangalore, India",
            "format": "json",
            "limit": 1
        }

        response = requests.get(url, params=params, headers={"User-Agent": "real-estate-app"})
        data = response.json()

        if len(data) == 0:
            raise Exception("Location not found")

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])

        return lat, lon

    except Exception as e:
        raise Exception(f"Geocoding error: {str(e)}")