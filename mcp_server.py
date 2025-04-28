# Add this near the top of your file, after the existing imports
from flask import Flask, request, jsonify, send_from_directory

#!/usr/bin/env python3
"""
mcp_server.py - Momentum Screener API Server

This Flask server provides a REST API interface to the momentum screener.
It allows client applications to run momentum screening with custom parameters
and retrieve the results.
"""
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import the screener modules
try:
    from momentum_screener_quick import quick_momentum_screener
    QUICK_MODE_AVAILABLE = True
except ImportError:
    QUICK_MODE_AVAILABLE = False
    print("Quick mode not available. Place momentum_screener_quick.py in the same directory.")

try:
    from momentum_screener_llm import momentum_screener
    FULL_MODE_AVAILABLE = True
except ImportError:
    FULL_MODE_AVAILABLE = False
    print("Full mode not available. Place momentum_screener_llm.py in the same directory.")

# Initialize Flask app
# app = Flask(__name__, static_folder='../client/build')
app = Flask(__name__) 
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://192.168.1.79:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})  # Enable CORS for all routes

@app.route('/')
def root():
    return """
    <html>
    <head><title>Momentum Screener API</title></head>
    <body>
        <h1>Momentum Screener API</h1>
        <p>API is running! Try these endpoints:</p>
        <ul>
            <li><a href="/api/status">/api/status</a></li>
            <li><a href="/api/universes">/api/universes</a></li>
        </ul>
    </body>
    </html>
    """

# Serve React static files
# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def serve(path):
#     if path != "" and os.path.exists(app.static_folder + '/' + path):
#         return send_from_directory(app.static_folder, path)
#     else:
#         return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check which screener modes are available"""
    return jsonify({
        'status': 'online',
        'quick_mode_available': QUICK_MODE_AVAILABLE,
        'full_mode_available': FULL_MODE_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/universes', methods=['GET'])
def get_universes():
    """Get available stock universes"""
    universes = [
        {'id': 0, 'name': 'S&P 500', 'description': 'Standard & Poor\'s 500 Index'},
        {'id': 1, 'name': 'S&P 1500', 'description': 'Combines the S&P 500, S&P MidCap 400, and S&P SmallCap 600'},
        {'id': 2, 'name': 'Russell 1000', 'description': 'Large-cap US stocks (CSV required)'},
        {'id': 3, 'name': 'Russell 3000', 'description': 'Top 3000 US stocks (CSV required)'},
        {'id': 4, 'name': 'TSX Composite', 'description': 'Toronto Stock Exchange Composite Index'},
        {'id': 5, 'name': 'Custom', 'description': 'Custom universe from text file'}
    ]
    return jsonify(universes)

@app.route('/api/screener/quick', methods=['POST'])
def run_quick_screener():
    """Run the quick version of the momentum screener"""
    if not QUICK_MODE_AVAILABLE:
        return jsonify({
            'error': 'Quick mode not available',
            'message': 'momentum_screener_quick.py is not available'
        }), 500
    
    # Get parameters from request
    data = request.json
    ticker_count = int(data.get('ticker_count', 30))
    lookback_days = int(data.get('lookback_days', 90))
    soft_breakout_pct = float(data.get('soft_breakout_pct', 0.005))
    proximity_threshold = float(data.get('proximity_threshold', 0.05))
    volume_threshold = float(data.get('volume_threshold', 1.2))
    
    # Run the screener
    try:
        result = quick_momentum_screener(
            ticker_count=ticker_count,
            lookback_days=lookback_days,
            soft_breakout_pct=soft_breakout_pct,
            proximity_threshold=proximity_threshold,
            volume_threshold=volume_threshold
        )
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error running quick screener'
        }), 500

@app.route('/api/screener/full', methods=['POST'])
def run_full_screener():
    """Run the full version of the momentum screener"""
    if not FULL_MODE_AVAILABLE:
        return jsonify({
            'error': 'Full mode not available',
            'message': 'momentum_screener_llm.py is not available'
        }), 500
    
    # Get parameters from request
    data = request.json
    universe_choice = int(data.get('universe_choice', 0))
    soft_breakout_pct = float(data.get('soft_breakout_pct', 0.005))
    proximity_threshold = float(data.get('proximity_threshold', 0.05))
    volume_threshold = float(data.get('volume_threshold', 1.2))
    lookback_days = int(data.get('lookback_days', 365))
    use_llm = bool(data.get('use_llm', False))
    
    # Run the screener
    try:
        result = momentum_screener(
            universe_choice=universe_choice,
            soft_breakout_pct=soft_breakout_pct,
            proximity_threshold=proximity_threshold,
            volume_threshold=volume_threshold,
            lookback_days=lookback_days
        )
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error running full screener'
        }), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get a list of available reports"""
    try:
        reports = []
        if os.path.exists('outputs'):
            for file in os.listdir('outputs'):
                if file.endswith('.xlsx'):
                    report_path = os.path.join('outputs', file)
                    report_time = datetime.fromtimestamp(os.path.getmtime(report_path))
                    reports.append({
                        'filename': file,
                        'path': report_path,
                        'created': report_time.isoformat(),
                        'size': os.path.getsize(report_path)
                    })
        return jsonify(reports)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error retrieving reports'
        }), 500

@app.route('/api/reports/<path:filename>', methods=['GET'])
def download_report(filename):
    """Download a specific report"""
    try:
        if os.path.exists('outputs') and os.path.exists(os.path.join('outputs', filename)):
            return send_from_directory('outputs', filename, as_attachment=True)
        else:
            return jsonify({
                'error': 'File not found',
                'message': f'Report {filename} does not exist'
            }), 404
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error downloading report'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


# Add this right after the CORS(app) line
