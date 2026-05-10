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

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) }
    : { r: 194, g: 239, b: 78 };
}

const Inference: React.FC = () => {
  const { selectedModel, selectedRun } = useGlobalState();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const drawPoseOnCanvas = (imgSrc: string, predictions: Prediction[], origW: number, origH: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Fit image within available container width, max 600px tall
      const maxW = containerRef.current?.clientWidth ?? 520;
      const maxH = 600;
      const scale = Math.min(maxW / origW, maxH / origH);
      const displayW = Math.round(origW * scale);
      const displayH = Math.round(origH * scale);

      canvas.width = displayW;
      canvas.height = displayH;

      // Draw image first
      ctx.drawImage(img, 0, 0, displayW, displayH);

      // Scale: original image px → canvas display px
      const sx = displayW / origW;
      const sy = displayH / origH;

      // Draw skeleton connections
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = Math.max(1.5, displayW / 150);
      for (const [i1, i2] of SKELETON_CONNECTIONS) {
        const j1 = predictions[i1];
        const j2 = predictions[i2];
        if (!j1 || !j2) continue;
        ctx.beginPath();
        ctx.moveTo(j1.x * sx, j1.y * sy);
        ctx.lineTo(j2.x * sx, j2.y * sy);
        ctx.stroke();
      }

      // Draw joint circles
      const radius = Math.max(4, displayW / 35);
      for (let i = 0; i < predictions.length; i++) {
        const pred = predictions[i];
        const color = JOINT_COLORS[i] ?? '#c2ef4e';
        const { r, g, b } = hexToRgb(color);

        ctx.beginPath();
        ctx.arc(pred.x * sx, pred.y * sy, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = Math.max(1.5, displayW / 250);
        ctx.stroke();
      }
    };
    img.src = imgSrc;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile || !previewUrl) return;

    setLoading(true);
    try {
      const data = await predictPose(selectedFile, selectedModel, selectedRun);
      setResult(data);
      drawPoseOnCanvas(
        previewUrl,
        data.predictions,
        data.original_size.width,
        data.original_size.height
      );
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } }; message?: string };
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

          {/* Image / Canvas display area */}
          <div
            ref={containerRef}
            style={{
              flex: 1,
              border: '2px dashed var(--border-purple)',
              borderRadius: '12px',
              minHeight: '400px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              overflow: 'hidden',
              background: 'rgba(0,0,0,0.2)',
            }}
          >
            <input
              type="file"
              id="file-upload"
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />

            {result ? (
              /* Canvas renders image + skeleton together — perfect alignment guaranteed */
              <canvas
                ref={canvasRef}
                style={{ display: 'block', maxWidth: '100%', borderRadius: '8px' }}
              />
            ) : previewUrl ? (
              <img
                src={previewUrl}
                alt="Preview"
                style={{ maxWidth: '100%', maxHeight: '580px', objectFit: 'contain', borderRadius: '8px', display: 'block' }}
              />
            ) : (
              <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '40px' }}>
                <Upload size={48} color="var(--accent-primary)" />
                <p className="text-uppercase" style={{ marginTop: '16px', fontWeight: '600' }}>Drop image here or click to upload</p>
              </label>
            )}

            {(previewUrl || result) && (
              <label
                htmlFor="file-upload"
                style={{
                  position: 'absolute', bottom: '8px', right: '8px',
                  background: 'rgba(0,0,0,0.6)', color: 'white', padding: '4px 10px',
                  borderRadius: '6px', fontSize: '0.75rem', cursor: 'pointer',
                  backdropFilter: 'blur(4px)', border: '1px solid rgba(255,255,255,0.2)',
                  zIndex: 2,
                }}
              >
                Change image
              </label>
            )}
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
