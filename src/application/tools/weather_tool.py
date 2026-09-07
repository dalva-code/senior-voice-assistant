import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict


class _CurrentWeather(BaseModel):
    model_config = ConfigDict(extra="ignore")

    temperature: float
    windspeed: float
    weathercode: int


class _WeatherResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    current_weather: _CurrentWeather


def _describe_weather_code(code: int) -> str:
    if code == 0:
        return "cielo despejado"
    if code in {1, 2, 3}:
        return "cielo parcialmente nuboso"
    if code in {45, 48}:
        return "niebla"
    if code in {51, 53, 55, 56, 57}:
        return "llovizna"
    if code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "lluvia"
    if code in {71, 73, 75, 77, 85, 86}:
        return "nieve"
    if code in {95, 96, 99}:
        return "tormenta"
    return "estado del cielo variable"


async def get_current_weather(lat: float, lon: float) -> str:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError("Coordenadas meteorológicas inválidas")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                },
            )
            response.raise_for_status()
        payload = _WeatherResponse.model_validate(response.json())
    except (httpx.HTTPError, ValueError):
        logger.bind(
            component="weather",
            event="request_failed",
            latitude=lat,
            longitude=lon,
        ).exception("No se pudo consultar Open-Meteo")
        return "No he podido consultar el tiempo ahora mismo. Inténtalo de nuevo más tarde."

    weather = payload.current_weather
    sky = _describe_weather_code(weather.weathercode)
    logger.bind(
        component="weather",
        event="current_weather",
        temperature=weather.temperature,
        windspeed=weather.windspeed,
        weathercode=weather.weathercode,
    ).info("Información meteorológica actualizada")
    return (
        f"Ahora mismo hay {weather.temperature:.1f} grados, {sky}, "
        f"y el viento sopla a {weather.windspeed:.1f} kilómetros por hora."
    )
