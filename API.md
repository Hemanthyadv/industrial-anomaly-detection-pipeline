# Industrial Sensor Inference API Reference

Base URL: `http://localhost:8000` (or `http://<host>:8000`)  
Interactive Swagger UI: `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/health` | Service health status, model load state, and cache statistics | None |
| `POST` | `/predict` | Single-sensor anomaly scoring | None |
| `POST` | `/predict-batch` | Batch anomaly scoring (1 to 1,000 items) | None |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint | None |

---

## 2. Endpoint Details

### `GET /health`
Returns the operational health and configuration state of the inference service.

#### Response `200 OK`:
```json
{
  "status": "ok",
  "model_a_loaded": true,
  "model_b_loaded": true,
  "model_version": "baseline",
  "cache_size": 42
}
```

---

### `POST /predict`
Evaluates a single multi-variable sensor observation for anomalous behavior.

#### Request Body (`application/json`):
```json
{
  "sensor_id": "sensor-0001",
  "value": 72.5,
  "unit": "C",
  "pressure": 2.05,
  "vibration": 0.048
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `sensor_id` | `string` | Yes | — | Unique sensor identifier (e.g. `sensor-0001`) |
| `value` | `float` | Yes | — | Primary reading (Temperature in °C, etc.) |
| `unit` | `string` | No | `"C"` | Measurement unit (`"C"`, `"bar"`, `"mm/s"`) |
| `pressure` | `float` | No | `2.0` | Coupled pressure reading in bar |
| `vibration` | `float` | No | `0.05` | Coupled vibration reading in mm/s |

#### Response `200 OK`:
```json
{
  "sensor_id": "sensor-0001",
  "anomaly_likelihood": 0.041289,
  "is_anomaly": false,
  "model_version": "baseline",
  "model_used": "model_a"
}
```

#### Response `422 Unprocessable Entity`:
Returned when input payload fails validation (e.g., missing required fields or invalid types).

#### Response `500 Internal Server Error`:
Returned when internal prediction pipeline fails unexpectedly.

---

### `POST /predict-batch`
Scores an array of sensor telemetry observations in a single round-trip.

#### Request Body (`application/json`):
```json
{
  "items": [
    {
      "sensor_id": "sensor-0001",
      "value": 72.5,
      "unit": "C",
      "pressure": 2.05,
      "vibration": 0.048
    },
    {
      "sensor_id": "sensor-0017",
      "value": 126.0,
      "unit": "C",
      "pressure": 4.20,
      "vibration": 0.250
    }
  ]
}
```

*Constraints:* `items` array must contain between `1` and `1,000` items.

#### Response `200 OK`:
```json
{
  "predictions": [
    {
      "sensor_id": "sensor-0001",
      "anomaly_likelihood": 0.041289,
      "is_anomaly": false,
      "model_version": "baseline",
      "model_used": "model_a"
    },
    {
      "sensor_id": "sensor-0017",
      "anomaly_likelihood": 0.984120,
      "is_anomaly": true,
      "model_version": "baseline",
      "model_used": "model_a"
    }
  ]
}
```

---

### `GET /metrics`
Standard Prometheus format metrics endpoint for scraping by Prometheus agent.

#### Example Response:
```text
# HELP prediction_requests_total Total prediction requests to the API
# TYPE prediction_requests_total counter
prediction_requests_total{endpoint="predict"} 1240.0
prediction_requests_total{endpoint="predict-batch"} 85.0
# HELP prediction_latency_seconds Prediction latency in seconds
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{endpoint="predict",le="0.005"} 1150.0
prediction_latency_seconds_bucket{endpoint="predict",le="0.01"} 1230.0
...
```

---

## 3. Example Client Usage

### Python `requests`:
```python
import requests

payload = {
    "sensor_id": "sensor-0001",
    "value": 72.5,
    "unit": "C",
    "pressure": 2.05,
    "vibration": 0.048,
}
response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

### cURL:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_id":"sensor-0001","value":72.5,"unit":"C","pressure":2.05,"vibration":0.048}'
```

