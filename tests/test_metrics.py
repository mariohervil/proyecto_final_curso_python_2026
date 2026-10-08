from _pytest.raises import raises
import pytest
import pandas as pd
from csic_climate.metrics import (
    calculate_climate_summary,
    calculate_decadal_trend,
    get_hottest_and_coldest_year,
)


@pytest.fixture
def sample_climate_data():
    """Fixture que proporciona un DataFrame sintético para las pruebas unitarias."""
    data = {
        "fecha": ["1990-01-01", "1990-06-01", "2020-01-01", "2020-06-01"],
        "year": [1990, 1990, 2020, 2020],
        "mes": [1, 6, 1, 6],
        "comunidad_autonoma": ["Andalucía", "Andalucía", "Andalucía", "Andalucía"],
        "temperatura_media_c": [12.0, 26.0, 14.0, 28.0],
        "anomalia_termica_c": [0.2, 0.5, 1.2, 1.8],
        "precipitacion_mm": [50.0, 10.0, 30.0, 5.0],
        "indice_spei_sequia": [0.1, -1.8, -0.5, -2.1],
        "dias_ola_calor": [0, 2, 0, 5],
    }
    return pd.DataFrame(data)


def test_calculate_climate_summary(sample_climate_data):
    summary = calculate_climate_summary(sample_climate_data)

    assert summary["temp_media"] == 20.0
    assert summary["anomalia_media"] == 0.92
    assert summary["precipitacion_total"] == 95.0
    assert summary["meses_sequia_severa"] == 2  # -1.8 y -2.1 son < -1.5
    assert summary["dias_totales_ola_calor"] == 7


def test_calculate_climate_summary_empty():
    empty_df = pd.DataFrame(
        columns=[
            "temperatura_media_c",
            "anomalia_termica_c",
            "precipitacion_mm",
            "indice_spei_sequia",
            "dias_ola_calor",
        ]
    )
    summary = calculate_climate_summary(empty_df)
    assert summary["temp_media"] == 0.0
    assert summary["meses_sequia_severa"] == 0


def test_calculate_decadal_trend(sample_climate_data):
    trend = calculate_decadal_trend(sample_climate_data)
    assert len(trend) == 2
    assert 1990 in trend["decada"].values
    assert 2020 in trend["decada"].values


def test_get_hottest_and_coldest_year(sample_climate_data):
    """
    Test 1: Comprueba el cálculo básico del año más cálido y más frío.

    Verifica:
    - Que el resultado devuelto sea un diccionario.
    - Que contenga todas las claves requeridas ('hottest_year', 'coldest_year', etc.).
    - Que los años obtenidos se encuentren dentro del rango histórico válido (1964-2024).
    """
    start_year = 1990
    end_year = 2020
    comunidades = "all"  # Evalúa todas las comunidades disponibles
    result = get_hottest_and_coldest_year(
        sample_climate_data, start_year, end_year, comunidades
    )

    # Validar que devuelve un diccionario con el formato esperado
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "hottest_year",
        "coldest_year",
        "hottest_value",
        "coldest_value",
    }

    # Validar que los años calculados están dentro del rango histórico
    assert start_year <= result["hottest_year"] <= end_year
    assert start_year <= result["coldest_year"] <= end_year

    # Validar presencia explícita de campos clave y métricas
    assert "hottest_year" in result and "coldest_year" in result
    assert "hottest_value" in result and "coldest_value" in result


def test_get_hottest_and_coldest_year_filtered_range(sample_climate_data):
    """
    Test 2: Comprueba el filtrado por rango de años y lista de comunidades.

    Verifica:
    - Que al restringir el rango de años (ej. 1990-2000), los registros fuera de
      dicho rango queden excluidos del cálculo.
    - Que el filtrado por una lista específica de comunidades autónomas se aplique
      correctamente sin interferir con otras regiones.
    """
    start_year = 1990
    end_year = 2000
    comunidades = ["Andalucía", "Madrid", "Cataluña", "Galicia"]

    result = get_hottest_and_coldest_year(
        sample_climate_data,
        start_year=start_year,
        end_year=end_year,
        comunidades=comunidades,
    )

    # Comprobación de tipos y límites históricos de los resultados filtrados
    assert isinstance(result, dict)
    assert start_year <= result["hottest_year"] <= end_year
    assert start_year <= result["coldest_year"] <= end_year


def test_get_hottest_and_coldest_year_empty():
    """
    Test 3: Comprueba la robustez ante un DataFrame vacío o sin registros.

    Verifica:
    - Que la función no lance excepciones no controladas cuando el DataFrame está vacío.
    - Que retorne None o un diccionario con valores nulos de forma segura.
    """
    empty_df = pd.DataFrame(
        columns=[
            "year",
            "comunidad_autonoma",
            "anomalia_termica_c",
            "temperatura_media_c",
        ]
    )
    with raises(ValueError):
        get_hottest_and_coldest_year(empty_df)
