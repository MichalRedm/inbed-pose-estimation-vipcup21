import React, { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, Download } from 'lucide-react';
import { predictPose } from '../services/api';

interface Prediction {
  joint: string;
  x: number;
  y: number;
}

interface InferenceResult {
  filename: string;
  original_size: { width: number; height: number };
  predictions: Prediction[];
}

const Inference: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const data = await predictPose(selectedFile);
      setResult(data);
    } catch (error) {
      console.error('Inference failed:', error);
      alert('Inference failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="inference-page">
      <div className="page-header">
        <h1 className="text-uppercase">Model Inference</h1>
        <p className="text-secondary">Run pose estimation on local images.</p>
      </div>

      <div className="inference-grid">
        <div className="glass card upload-section">
          <div className={`dropzone ${selectedFile ? 'has-file' : ''}`}>
            <input type="file" id="file-upload" onChange={handleFileChange} accept="image/*" />
            <label htmlFor="file-upload">
              {previewUrl ? (
                <div className="preview-container">
                  <img ref={imageRef} src={previewUrl} alt="Preview" className="image-preview" />
                  {result && (
                    <svg 
                      className="pose-overlay"
                      viewBox={`0 0 ${result.original_size.width} ${result.original_size.height}`}
                      preserveAspectRatio="xMidYMid meet"
                    >
                      {result.predictions.map((pred, i) => (
                        <g key={i}>
                          <circle
                            cx={pred.x}
                            cy={pred.y}
                            r={result.original_size.width / 100} // Dynamic radius based on image size
                            fill="var(--accent-lime)"
                            stroke="white"
                            strokeWidth={result.original_size.width / 400}
                          />
                        </g>
                      ))}
                    </svg>
                  )}
                </div>
              ) : (
                <div className="upload-placeholder">
                  <Upload size={48} color="var(--accent-primary)" />
                  <p className="text-uppercase" style={{ marginTop: '16px', fontWeight: '600' }}>Drop image here or click to upload</p>
                </div>
              )}
            </label>
          </div>
          
          <div className="upload-actions">
            <button 
              className="btn-lime" 
              onClick={handleRunInference} 
              disabled={!selectedFile || loading}
              style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
            >
              {loading ? <RefreshCw className="spin" size={18} /> : <Play size={18} />}
              Run Inference
            </button>
          </div>
        </div>

        <div className="glass card results-section">
          <h3 className="text-uppercase micro-label" style={{ marginBottom: '16px' }}>Predictions</h3>
          {result ? (
            <div className="predictions-list">
              <table className="results-table">
                <thead>
                  <tr className="text-uppercase">
                    <th>Joint</th>
                    <th>X</th>
                    <th>Y</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.map((pred, i) => (
                    <tr key={i}>
                      <td>{pred.joint}</td>
                      <td>{pred.x.toFixed(1)}</td>
                      <td>{pred.y.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button className="btn-primary" style={{ marginTop: '24px', width: '100%' }}>
                <Download size={18} />
                Export Results
              </button>
            </div>
          ) : (
            <div className="empty-state">
              <p className="text-secondary">Run inference to see results.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Inference;
