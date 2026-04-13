# BodyScanner Backend

FastAPI service that accepts a body-scan video from the iOS app (or the bizbozz React widget), runs HMR2.0 inference, and returns measurements + an OBJ mesh.

## Architecture

```
iOS app / React widget
        │  POST /contacts/{contactProfileId}/analyze  (multipart video + Bearer JWT)
        ▼
  BodyScanner API  (this repo)
        │  verifies bizbozz JWT (JWT_SECRET_BUSINESS)
        │  runs HMR2.0 / 4D-Humans inference
        │  saves scan to PostgreSQL
        │  stores OBJ mesh on disk (or S3)
        ▼
  Response: measurements JSON + meshUrl
```

### Data model

```
bizbozz:   User ──< business_persona >── Business ──< ContactProfile
                                                            │
BodyScanner:                                             Scan ◄── stored here
```

A scan is always owned by a **ContactProfile** (the end customer / gym member).
`business_id` is stored redundantly for fast access control.

## Setup

### 1. Clone and configure

```bash
git clone <this-repo> bodyscanner-backend
cd bodyscanner-backend
cp .env.example .env
# Edit .env — set JWT_SECRET_BUSINESS from bizbozz team
```

### 2. Install Python deps

Requires **Python 3.11+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install 4D-Humans (HMR2.0) from GitHub
pip install git+https://github.com/shubham-goel/4D-Humans.git
```

> **No GPU?** The server still runs and returns mock measurements.
> HMR2.0 is only invoked when `torch` + `hmr2` are importable.

### 3. Download HMR2.0 checkpoint

```bash
python -m hmr2.utils.download_models --download_dir ./checkpoints
# Sets HMR2_CHECKPOINT=./checkpoints/hmr2.0a.ckpt in .env
```

### 4. Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

### 5. Run with Docker (GPU host)

```bash
docker build -t bodyscanner-api .
docker run -p 8000:8000 \
  --gpus all \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/meshes:/app/meshes \
  -v $(pwd)/checkpoints:/app/checkpoints \
  bodyscanner-api
```

## API reference

### Authentication

All endpoints require a valid bizbozz JWT in the `Authorization: Bearer <token>` header.

The JWT is obtained by the client (iOS app or React widget) via bizbozz Firebase auth and forwarded directly to this service.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/contacts/{contactProfileId}/analyze` | Upload video, run inference, save scan |
| `GET`  | `/contacts/{contactProfileId}/scans` | List scans for a contact |
| `GET`  | `/scans/{scanId}` | Get scan details |
| `GET`  | `/scans/{scanId}/mesh` | Download OBJ mesh |
| `GET`  | `/scans/{scanId}/thumbnail` | Download JPEG thumbnail |
| `DELETE` | `/scans/{scanId}` | Delete scan |
| `GET`  | `/health` | Health check |

### POST `/contacts/{contactProfileId}/analyze`

**Request** — multipart/form-data:
- `video` (file): `.mp4` or `.mov`, max 200 MB

**Response** — JSON:
```json
{
  "scanId": "uuid",
  "contactProfileId": "uuid",
  "measurements": {
    "height_cm": 175.5,
    "chest_cm": 96.2,
    "waist_cm": 80.4,
    "hip_cm": 98.1,
    "shoulder_width_cm": 44.3,
    "accuracy": 0.87
  },
  "meshUrl": "/scans/{scanId}/mesh",
  "thumbnailUrl": "/scans/{scanId}/thumbnail"
}
```

## iOS integration

In `UploadView.swift`, replace the mock with:

```swift
func uploadAndAnalyze(videoURL: URL, contactProfileId: String, token: String) async throws -> AnalyzeResponse {
    let url = URL(string: "https://your-backend.com/contacts/\(contactProfileId)/analyze")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"video\"; filename=\"scan.mp4\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
    body.append(try Data(contentsOf: videoURL))
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body

    let (data, _) = try await URLSession.shared.data(for: request)
    return try JSONDecoder().decode(AnalyzeResponse.self, from: data)
}
```

## bizbozz React widget integration

Install the npm package (separate repo):

```bash
npm install @gymheros/bodyscanner-widget
```

```tsx
import { BodyScannerWidget } from '@gymheros/bodyscanner-widget';

<BodyScannerWidget
  contactProfileId={contact.id}
  businessId={currentBusiness.id}
  bearerToken={bizbozzJwt}
  apiBaseUrl="https://your-backend.com"
/>
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `JWT_SECRET_BUSINESS` | ✅ | From bizbozz team — must match bizbozz JWT signing secret |
| `DATABASE_URL` | ✅ | SQLAlchemy async URL |
| `MESH_STORAGE_PATH` | ✅ | Directory (or S3 prefix) for OBJ/thumbnail files |
| `HMR2_CHECKPOINT` | — | Path to `hmr2.0a.ckpt`; falls back to mock if not set |
| `MAX_VIDEO_SIZE_MB` | — | Default 200 |
