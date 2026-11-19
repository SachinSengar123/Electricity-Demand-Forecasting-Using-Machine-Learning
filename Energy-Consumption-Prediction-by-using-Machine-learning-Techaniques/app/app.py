"""
GridCast AI - Flask Web Application
Provides full REST API endpoints:
- POST /api/upload-dataset
- POST /api/train-model
- GET /api/forecast
- GET /api/peak-analysis
- GET /api/model-performance
- GET /api/dataset-info
- GET /api/overview
- GET /api/status
- GET /api/forecast/download
- POST /api/reset-dataset
"""

import os
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from model_service import ForecastEngine, get_current_ist_time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

engine = ForecastEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'online',
        'app_name': 'GridCast AI',
        'version': '2.5.0',
        'timezone': 'Asia/Kolkata (IST)',
        'current_time_ist': get_current_ist_time().strftime('%d %B %Y, %I:%M:%S %p IST'),
        'is_demo_data': not engine.is_custom_data
    })

@app.route('/api/overview', methods=['GET'])
def get_overview():
    model_name = request.args.get('model', 'LSTM')
    temp_mod = float(request.args.get('temp_modifier', 0.0))
    forecast_data = engine.forecast_next_24h(model_name, temp_modifier=temp_mod)
    
    return jsonify({
        'kpis': forecast_data['summary'],
        'is_demo_data': forecast_data['is_demo_data'],
        'model_used': forecast_data['model_used'],
        'recent_history': forecast_data['historical_24h'],
        'forecast_trend': forecast_data['forecast'],
        'alerts': forecast_data['alerts'][:2],
        'system_metadata': {
            'last_dataset_update': engine.quality_report.get('date_range', 'Continuous Sample Feed'),
            'last_forecast_time': forecast_data['generated_at'],
            'last_trained_time': engine.last_trained_time,
            'dataset_status': 'Validated (Active)' if not engine.is_custom_data else 'Custom User CSV Ingested',
            'model_status': 'Trained & Operational'
        }
    })

# Specified endpoint: GET /api/forecast
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    model_name = request.args.get('model', 'LSTM')
    temp_mod = float(request.args.get('temp_modifier', 0.0))
    crit_pct = float(request.args.get('crit_thresh', 1.15))
    high_pct = float(request.args.get('high_thresh', 1.05))
    low_pct = float(request.args.get('low_thresh', 0.90))
    
    res = engine.forecast_next_24h(
        selected_model=model_name,
        temp_modifier=temp_mod,
        critical_threshold_pct=crit_pct,
        high_threshold_pct=high_pct,
        low_threshold_pct=low_pct
    )
    return jsonify(res)

# Specified endpoint: GET /api/peak-analysis
@app.route('/api/peak-analysis', methods=['GET'])
def get_peak_analysis():
    model_name = request.args.get('model', 'LSTM')
    temp_mod = float(request.args.get('temp_modifier', 0.0))
    crit_pct = float(request.args.get('crit_thresh', 1.15))
    high_pct = float(request.args.get('high_thresh', 1.05))
    low_pct = float(request.args.get('low_thresh', 0.90))
    
    res = engine.forecast_next_24h(
        selected_model=model_name,
        temp_modifier=temp_mod,
        critical_threshold_pct=crit_pct,
        high_threshold_pct=high_pct,
        low_threshold_pct=low_pct
    )
    return jsonify({
        'is_demo_data': res['is_demo_data'],
        'summary': res['summary'],
        'hourly_patterns': res['forecast'],
        'breakdown': res['breakdown'],
        'alerts': res['alerts'],
        'thresholds_used': {
            'critical_pct': crit_pct,
            'high_pct': high_pct,
            'low_pct': low_pct
        }
    })

# Specified endpoint: GET /api/model-performance
@app.route('/api/model-performance', methods=['GET'])
def get_model_performance():
    return jsonify({
        'is_demo_data': not engine.is_custom_data,
        'best_model': engine.best_model_name,
        'metrics': engine.metrics,
        'test_eval': engine.test_eval_data,
        'last_trained_time': engine.last_trained_time
    })

# Specified endpoint: GET /api/dataset-info
@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info_endpoint():
    res = engine.get_dataset_info()
    return jsonify(res)

# Specified endpoint: POST /api/upload-dataset
@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset_endpoint():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part provided in the request.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected for upload.'}), 400
        
    try:
        df_upload = pd.read_csv(file)
        
        # Validation
        req_cols = ['Date_Time', 'Electricity_Demand']
        missing = [c for c in req_cols if c not in df_upload.columns]
        if missing:
            return jsonify({
                'success': False,
                'error': f'CSV validation failed. Missing required columns: {", ".join(missing)}. Required columns: Date_Time, Electricity_Demand (Optional: Temperature, Humidity)'
            }), 400
            
        if len(df_upload) < 24:
            return jsonify({
                'success': False,
                'error': f'Dataset contains only {len(df_upload)} records. At least 24 hourly rows are required for time-series feature engineering.'
            }), 400
            
        # Retrain ML models
        engine.load_and_train(df_upload)
        
        return jsonify({
            'success': True,
            'message': f'Dataset uploaded successfully! Ingested {len(df_upload)} records. Models retrained and forecasts updated.',
            'dataset_info': engine.get_dataset_info()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to parse CSV: {str(e)}'}), 500

# Specified endpoint: POST /api/train-model
@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    try:
        # Re-train models with existing active dataset
        engine.load_and_train(engine.df if engine.is_custom_data else None)
        return jsonify({
            'success': True,
            'message': 'Models successfully re-trained on current dataset slice.',
            'best_model': engine.best_model_name,
            'metrics': engine.metrics,
            'last_trained_time': engine.last_trained_time
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Model training failed: {str(e)}'}), 500

@app.route('/api/reset-dataset', methods=['POST'])
def reset_dataset_endpoint():
    engine.load_and_train(None)
    return jsonify({
        'success': True,
        'message': 'Reset to dynamic demonstration dataset aligned with live system IST clock.',
        'dataset_info': engine.get_dataset_info()
    })

@app.route('/api/forecast/download', methods=['GET'])
def download_forecast_csv():
    model_name = request.args.get('model', 'LSTM')
    temp_mod = float(request.args.get('temp_modifier', 0.0))
    res = engine.forecast_next_24h(model_name, temp_modifier=temp_mod)
    
    df_out = pd.DataFrame(res['forecast'])
    output = io.StringIO()
    df_out[['timestamp', 'display_time', 'predicted_demand', 'upper_bound', 'lower_bound', 'status']].to_csv(output, index=False)
    
    filename = f"GridCast_24h_Forecast_{model_name.replace(' ', '_')}_{get_current_ist_time().strftime('%Y%m%d_%H%M')}_IST.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"GridCast AI dashboard running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
