from pathlib import Path
import urllib.request

#Link de descarga del dataset OPSD
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"


def download_opsd(out_path: str = "data/raw/opsd_time_series_60min_singleindex.csv") -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        print(f"El conjunto de datos ya existe: {out}")
        return str(out)

    print("Descargando dataset...")
    urllib.request.urlretrieve(OPSD_URL, out)
    print(f"El dataset se ha descargado correctamente:{out}")
    return str(out)


if __name__ == "__main__":
    download_opsd()