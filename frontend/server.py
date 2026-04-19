import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Load the file into memory once (this will take a moment but saves reading it every request)
cache = {}

class ChunkedReplayHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/replay':
            qs = parse_qs(parsed.query)
            filename = qs.get('file', [''])[0]
            start = int(qs.get('start', ['0'])[0])
            size = int(qs.get('size', ['100'])[0])
            
            filepath = os.path.join('public', 'replays', filename)
            
            if not os.path.exists(filepath):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'File not found')
                return

            # Lazy load into memory
            if filename not in cache:
                print(f"Loading {filename} into memory... this might take a minute.")
                with open(filepath, 'r') as f:
                    cache[filename] = json.load(f)
                print("Loaded!")

            data = cache[filename]
            chunk = data[start:start+size]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps({
                "totalFrames": len(data),
                "frames": chunk
            }).encode('utf-8'))
        else:
            super().do_GET()

if __name__ == '__main__':
    print("Starting server on http://localhost:8000")
    HTTPServer(('localhost', 8000), ChunkedReplayHandler).serve_forever()
