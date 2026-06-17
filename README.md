# Maize Leaf Disease Detection — Flask Backend

Backend API for the *Maize Leaf Disease Detection Using CNN* major project
(Pokhara University). Implements the REST API described in proposal
sections 3.5, 3.10, and 3.11: a Flask server that loads the trained
MobileNetV2-based CNN once at startup, classifies uploaded maize leaf
images into 10 classes, and returns disease info, confidence score,
severity level, and treatment/prevention advice for the Flutter app.

## 1. Folder structure

```
maize-backend/
├── app.py                      # Main Flask app, /predict endpoint
├── requirements.txt
├── render.yaml                 # Render.com deployment config
├── Procfile                    # Backup start command for Render
├── test_api.py                 # Manual test script
├── model/
│   └── maize_model.h5          # <-- PUT YOUR TRAINED MODEL HERE
└── utils/
    ├── preprocessing.py        # Image resize/normalize logic
    └── recommendations.py      # Disease knowledge base (section 3.10)
```

## 2. One-time setup

```bash
cd maize-backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Copy your trained `.h5` model file into `model/maize_model.h5`. See
`model/PLACE_MODEL_HERE.txt` for two important checks before running.

## 3. Before running — verify these two things

These are the two most common reasons a model that performed well in
training gives wrong predictions once deployed behind an API:

1. **Class order.** Open `app.py` and check the `CLASS_NAMES` list. It
   must be in the *exact same order* Keras assigned during training. In
   your training notebook, run:
   ```python
   print(train_generator.class_indices)
   ```
   and reorder `CLASS_NAMES` in `app.py` to match those indices exactly
   (index 0 first, index 1 second, and so on).

2. **Preprocessing.** Open `utils/preprocessing.py`. It currently divides
   pixel values by 255.0 (matches proposal section 3.7). If your training
   pipeline instead used
   `tf.keras.applications.mobilenet_v2.preprocess_input` (which scales to
   `[-1, 1]`), swap that in — whatever was used during training must be
   replicated exactly at inference time.

## 4. Run locally

```bash
python app.py
```

You should see:
```
Loading model from model/maize_model.h5 ...
Model loaded successfully.
 * Running on http://0.0.0.0:5000
```

Visit `http://127.0.0.1:5000/` in a browser — you should get a JSON
health check response confirming the server and model are up.

## 5. Test the /predict endpoint

Using the included test script:
```bash
python test_api.py path/to/some_leaf_image.jpg
```

Or with curl:
```bash
curl -X POST -F "image=@path/to/leaf.jpg" http://127.0.0.1:5000/predict
```

Or with Postman (as listed in your proposal's tool list, section 1.7.1):
- Method: `POST`
- URL: `http://127.0.0.1:5000/predict`
- Body type: `form-data`
- Key: `image`, type `File`, value: choose your test image

Expected response shape:
```json
{
  "disease": "Common Rust",
  "confidence": 94.32,
  "is_healthy": false,
  "severity": "Yellow",
  "description": "Caused by the fungus Puccinia sorghi...",
  "treatment": "Apply a triazole or strobilurin-based fungicide...",
  "prevention": "Plant resistant/tolerant maize varieties...",
  "all_class_probabilities": {
    "Healthy": 1.2,
    "Common Rust": 94.32,
    "...": "..."
  }
}
```

## 6. Connecting from the Flutter app

In your Flutter `Request Handling Module` (proposal section 3.5), send a
multipart POST request to `/predict` with the field name `image`. Example
using the `http` package:

```dart
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> predictDisease(File imageFile) async {
  final uri = Uri.parse('http://<YOUR_SERVER_URL>/predict');
  final request = http.MultipartRequest('POST', uri);
  request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

  final streamedResponse = await request.send();
  final response = await http.Response.fromStream(streamedResponse);

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Prediction failed: ${response.body}');
  }
}
```

While testing locally with an Android emulator, use `http://10.0.2.2:5000`
instead of `127.0.0.1` (emulator's alias for your host machine). On a
physical device on the same Wi-Fi, use your computer's local IP address,
e.g. `http://192.168.1.X:5000`.

## 7. Deploying to Render.com (proposal section 3.11)

1. Push this `maize-backend` folder to a GitHub repo (or a `backend/`
   subfolder inside your existing
   `olitejendra/Maize-Leaf-Disease-Detection` repo — Render lets you set a
   root directory).
2. **Important:** Git has a 100MB file size limit by default. If your
   `.h5` model file is large, you'll need Git LFS:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   ```
3. Go to [render.com](https://render.com) → New → Web Service → connect
   your GitHub repo.
4. Render should auto-detect `render.yaml`. If not, set manually:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Deploy. Render gives you a public URL like
   `https://maize-disease-detection-api.onrender.com`.
6. Update the Flutter app's API base URL to that Render URL.

Note: Render's free tier spins down on inactivity, so the first request
after idle time will be slow (cold start, often 30–60s) while TensorFlow
loads the model again. This is expected behavior on the free plan, not a
bug.

## 8. Next steps to match the rest of your proposal

- **Firebase (section 3.13):** handled entirely on the Flutter side — this
  backend is stateless and does not store prediction history (the
  proposal specifies the Flutter client persists history directly to
  Firestore).
- **Evaluation metrics (section 3.14):** these are computed during model
  training/evaluation in your Jupyter notebook (confusion matrix,
  accuracy, recall, specificity, F1), not part of this backend code.
