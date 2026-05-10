import React, { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, Download, Layers } from 'lucide-react';
import { predictPose } from '../services/api';
import { useGlobalState } from '../context/GlobalStateContext';

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

const SKELETON_CONNECTIONS = [
  [13, 12], // Head - Thorax
  [12, 8],  [12, 9], // Thorax - Shoulders
  [8, 7],   [7, 6],  // Right Arm
  [9, 10],  [10, 11], // Left Arm
  [8, 2],   [9, 3],  // Torso
  [2, 3],            // Pelvis
  [2, 1],   [1, 0],  // Right Leg
  [3, 4],   [4, 5]   // Left Leg
];

const JOINT_COLORS: Record<number, string> = {
  13: '#fa7faa', // Head
  12: '#ffb287', // Thorax
  8: '#c2ef4e', 7: '#c2ef4e', 6: '#c2ef4e', // Right arm
  9: '#6a5fc1', 10: '#6a5fc1', 11: '#6a5fc1', // Left arm
  2: '#fa7faa', 1: '#fa7faa', 0: '#fa7faa', // Right leg
  3: '#6a5fc1', 4: '#6a5fc1', 5: '#6a5fc1' // Left leg
};

const Inference: React.FC = () => {
  const { selectedModel, selectedRun } = useGlobalState();
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
      const data = await predictPose(selectedFile, selectedModel, selectedRun);
      setResult(data);
    } catch (error: any) {
      console.error('Inference failed:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Inference failed. Is the backend running?';
      alert(`Inference failed: ${errorMsg}`);
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
          
          <div style={{ marginBottom: '20px', padding: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', border: '1px solid var(--border-purple)' }}>
            <div className="micro-label text-secondary" style={{ marginBottom: '8px' }}>Active Model</div>
            {selectedRun ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <RefreshCw size={14} className="text-accent-pink" />
                <span>Run: {selectedRun}</span>
              </div>
            ) : selectedModel ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={14} className="text-accent-pink" />
                <span>{selectedModel}</span>
              </div>
            ) : (
              <div className="text-warning" style={{ fontSize: '0.85rem' }}>No model selected (choose from header)</div>
            )}
          </div>

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
                      {/* Connections */}
                      {SKELETON_CONNECTIONS.map(([idx1, idx2], i) => {
                        const j1 = result.predictions[idx1];
                        const j2 = result.predictions[idx2];
                        if (!j1 || !j2) return null;
                        return (
                          <line
                            key={`line-${i}`}
                            x1={j1.x} y1={j1.y}
                            x2={j2.x} y2={j2.y}
                            stroke="white"
                            strokeWidth={result.original_size.width / 300}
                            strokeOpacity="0.5"
                          />
                        );
                      })}

                      {/* Joints */}
                      {result.predictions.map((pred, i) => (
                        <g key={i}>
                          <circle
                            cx={pred.x}
                            cy={pred.y}
                            r={result.original_size.width / 100}
                            fill={JOINT_COLORS[i] || 'var(--accent-lime)'}
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
              disabled={!selectedFile || loading || (!selectedModel && !selectedRun)}
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
