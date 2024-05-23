import requests

# server url
URL = "http://127.0.0.1:5000/predict"

# audio file we'd like to send for predicting keyword
wav_path = r"C:\Users\LENOVO\Desktop\speech_recognition\MLTU\Datasets\LJSpeech-1.1\wavs\LJ001-0001.wav"

if __name__ == "__main__":

    # open file
    with open(wav_path, "rb") as file:
        content = file.read()
        # package stuff to send and perform POST request
        values = {"file": (wav_path, file, "audio/wav")}
        response = requests.post(URL, files=values)
        

    print("Response status code:", response.status_code)
    print("Response headers:", response.headers)
    print("Response content:", response.content)

    # Try to parse JSON and print the prediction
    try:
        data = response.json()
        print("Predicted keyword: {}".format(data["prediction"]))
    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not in JSON format")
    except KeyError:
        print("Error: 'prediction' key not found in the response")