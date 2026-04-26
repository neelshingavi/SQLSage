"""OpenEnv-compatible server app entrypoint."""

from sqlsage.app import app


def main():
    return app


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
