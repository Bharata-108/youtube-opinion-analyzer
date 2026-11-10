"""HTTP REST API and Static Web Server for YouTube Opinion Analysis System.

Provides high-performance endpoints for Narrow Mode, Vast Mode, CSV uploads,
and serves the modern HTML5/CSS/JavaScript dashboard.
Built entirely with Python's standard library for zero-overhead portability.
"""

import os
import sys
import json
import mimetypes
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure local packages are in path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import SAMPLE_DATA_DIR, DEFAULT_NUM_CLUSTERS, YOUTUBE_API_KEY
from core.collector import YouTubeDataCollector
from core.pipeline import execute_pipeline


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
STATIC_DIR = BASE_DIR / "static"


class OpinionAnalysisRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the Opinion Analysis API and Dashboard."""

    def _send_json_response(self, data: dict, status_code: int = 200):
        """Helper to send JSON response with appropriate headers."""
        response_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        """Handles CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handles static file serving and health endpoints."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # 1. API: Health Check
        if path == "/api/health":
            self._send_json_response({"status": "ready", "service": "YouTube Opinion Analysis System"})
            return

        # 2. API: Sample Datasets list
        if path == "/api/sample-datasets":
            datasets = []
            if SAMPLE_DATA_DIR.exists():
                for f in SAMPLE_DATA_DIR.glob("*.csv"):
                    datasets.append({
                        "filename": f.name,
                        "name": f.stem.replace("_", " ").title(),
                        "path": str(f)
                    })
            self._send_json_response({"datasets": datasets})
            return

        # 3. Serve Frontend (index.html)
        if path in ("/", "/index.html"):
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Frontend file index.html not found.")
                return

        # 4. Serve Static Files
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = STATIC_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or "application/octet-stream"
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        self.send_error(404, f"Resource {path} not found.")

    def do_POST(self):
        """Handles analysis requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            self._send_json_response({"status": "error", "message": "Malformed JSON payload."}, 400)
            return

        # 1. Narrow Mode Analysis
        if path == "/api/analyze/narrow":
            use_demo = payload.get("use_demo", True)
            api_key = payload.get("api_key", "").strip() or YOUTUBE_API_KEY
            video_url = payload.get("video_url", "").strip()
            k_clusters = int(payload.get("clusters", DEFAULT_NUM_CLUSTERS))

            try:
                if use_demo or not video_url:
                    sample_file = SAMPLE_DATA_DIR / "movie_trailer_comments.csv"
                    raw_comments = YouTubeDataCollector.load_from_csv(sample_file)
                    source_title = "Chronicles of Eldoria - Official Movie Trailer"
                elif not api_key:
                    self._send_json_response({
                        "status": "error",
                        "message": "No YouTube API Key found. Set YOUTUBE_API_KEY in server environment or provide a key in Advanced Settings."
                    }, 400)
                    return
                else:
                    collector = YouTubeDataCollector(api_key=api_key)
                    vid_id = collector.extract_video_id(video_url)
                    if not vid_id:
                        self._send_json_response({"status": "error", "message": "Invalid YouTube URL or Video ID provided."}, 400)
                        return
                    raw_comments = collector.fetch_narrow_comments(vid_id, max_comments=250)
                    source_title = f"Live YouTube Video ({vid_id})"

                result = execute_pipeline(
                    raw_comments=raw_comments,
                    mode="Narrow",
                    source_name=source_title,
                    n_clusters=k_clusters,
                    enable_suspicious=True,
                    enable_overlap=False
                )
                self._send_json_response(result)

            except Exception as e:
                self._send_json_response({"status": "error", "message": str(e)}, 500)
            return

        # 2. Vast Mode Analysis
        if path == "/api/analyze/vast":
            use_demo = payload.get("use_demo", True)
            api_key = payload.get("api_key", "").strip() or YOUTUBE_API_KEY
            queries = payload.get("queries", [])
            k_clusters = int(payload.get("clusters", DEFAULT_NUM_CLUSTERS))

            try:
                if use_demo or not queries:
                    sample_file = SAMPLE_DATA_DIR / "tech_launch_comments.csv"
                    raw_comments = YouTubeDataCollector.load_from_csv(sample_file)
                    source_title = "Tech Launch Flagship Comparisons"
                elif not api_key:
                    self._send_json_response({
                        "status": "error",
                        "message": "No YouTube API Key found. Set YOUTUBE_API_KEY in server environment or provide a key in Advanced Settings."
                    }, 400)
                    return
                else:
                    collector = YouTubeDataCollector(api_key=api_key)
                    raw_comments = collector.fetch_vast_comments(queries, videos_per_query=2, comments_per_video=75)
                    source_title = f"Vast Search: {', '.join(queries)}"

                result = execute_pipeline(
                    raw_comments=raw_comments,
                    mode="Vast",
                    source_name=source_title,
                    n_clusters=k_clusters,
                    enable_suspicious=True,
                    enable_overlap=True
                )
                self._send_json_response(result)

            except Exception as e:
                self._send_json_response({"status": "error", "message": str(e)}, 500)
            return

        # 3. Custom Raw CSV Comments Analysis
        if path == "/api/analyze/custom":
            csv_text = payload.get("csv_text", "")
            mode = payload.get("mode", "Narrow")
            k_clusters = int(payload.get("clusters", DEFAULT_NUM_CLUSTERS))

            if not csv_text:
                self._send_json_response({"status": "error", "message": "Empty CSV content received."}, 400)
                return

            try:
                import io
                raw_comments = YouTubeDataCollector.load_from_csv(io.StringIO(csv_text))
                result = execute_pipeline(
                    raw_comments=raw_comments,
                    mode=mode,
                    source_name="User Uploaded Dataset",
                    n_clusters=k_clusters,
                    enable_suspicious=True,
                    enable_overlap=(mode == "Vast")
                )
                self._send_json_response(result)
            except Exception as e:
                self._send_json_response({"status": "error", "message": str(e)}, 500)
            return

        self.send_error(404, f"API endpoint {path} not found.")


def run_server(port: int = PORT):
    """Starts the HTTP server."""
    server_address = (HOST, port)
    httpd = HTTPServer(server_address, OpinionAnalysisRequestHandler)
    print(f"================================================================")
    print(f"  YouTube Opinion Analysis System - Web Server Running")
    print(f"  Access Dashboard at: http://localhost:{port}/")
    print(f"================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port_arg)
