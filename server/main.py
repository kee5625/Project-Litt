import uvicorn

from app import app
from config import PORT


def main() -> None:
    try:
        port_value = int(PORT)
    except ValueError:
        port_value = 8000

    uvicorn.run(app, host="0.0.0.0", port=port_value)


if __name__ == "__main__":
    main()
