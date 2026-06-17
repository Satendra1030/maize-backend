"""
Quick manual test script for the /predict endpoint.

Usage:
    python test_api.py path/to/leaf_image.jpg

Make sure the Flask server is already running (python app.py) before
running this script.
"""

import sys
import requests


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    url = "http://127.0.0.1:5000/predict"

    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, files=files)

    print("Status code:", response.status_code)
    print("Response JSON:")
    print(response.json())


if __name__ == "__main__":
    main()
