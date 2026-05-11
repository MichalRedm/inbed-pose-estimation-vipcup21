import React, { useState, useRef, useEffect } from 'react';
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

interface RunInferenceProps {
  runId: string;
}

const RunInference: React.FC<RunInferenceProps> = ({ runId }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!result || !previewUrl) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { predictions, original_size: { width: origW, height: origH } } = result;

    const img = new Image();
    img.onload = () => {
      const maxW = containerRef.current?.clientWidth ?? 520;
      const maxH = 400;
      const scale = Math.min(maxW / origW, maxH / origH);
      const displayW = Math.round(origW * scale);
      const displayH = Math.round(origH * scale);

      canvas.width = displayW;
      canvas.height = displayH;
      ctx.drawImage(img, 0, 0, displayW, displayH);

      const sx = displayW / origW;
      const sy = displayH / origH;

      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = 2;
      for (const [i1, i2] of SKELETON_CONNECTIONS) {
        const j1 = predictions[i1];
        const j2 = predictions[i2];
        if (!j1 || !j2) continue;
        ctx.beginPath();
        ctx.moveTo(j1.x * sx, j1.y * sy);
        ctx.lineTo(j2.x * sx, j2.y * sy);
        ctx.stroke();
      }

      for (let i = 0; i < predictions.length; i++) {
        const pred = predictions[i];
        ctx.beginPath();
        ctx.arc(pred.x * sx, pred.y * sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = JOINT_COLORS[i] ?? '#c2ef4e';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    };
    img.src = previewUrl;
  }, [result, previewUrl]);

  const handleRunInference = async () => {
    if (!selectedFile) return;
    setLoading(true);
    try {
      const data = await predictPose(selectedFile, undefined, runId);
      setResult(data);
    } catch (err) {
      console.error('Inference failed:', err);
      alert('Inference failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="run-inference grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '20px', height: '100%' }}>
      <div className="flex-column" style={{ gap: '20px' }}>
        <div 
          ref={containerRef}
          className="glass card" 
          style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', background: 'rgba(0,0,0,0.2)', border: '2px dashed var(--border-purple)' }}
        >
          <input type="file" id="run-inference-upload" hidden onChange={(e) => {
            if (e.target.files?.[0]) {
              setSelectedFile(e.target.files[0]);
              setPreviewUrl(URL.createObjectURL(e.target.files[0]));
              setResult(null);
            }
          }} />
          
          <canvas ref={canvasRef} style={{ display: result ? 'block' : 'none', maxWidth: '100%', maxHeight: '100%' }} />
          {!result && previewUrl && <img src={previewUrl} style={{ maxWidth: '100%', maxHeight: '400px' }} />}
          {!previewUrl && (
            <label htmlFor="run-inference-upload" style={{ cursor: 'pointer', textAlign: 'center' }}>
              <Upload size={40} color="var(--accent-primary)" />
              <p className="micro-label">Click to upload image</p>
            </label>
          )}
          
          {(previewUrl || result) && (
            <label htmlFor="run-inference-upload" style={{ position: 'absolute', top: 12, right: 12, background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.7rem' }}>Change</label>
          )}
        </div>

        <button className="btn-lime" disabled={!selectedFile || loading} onClick={handleRunInference} style={{ width: '100%' }}>
          {loading ? <RefreshCw className="spin" size={18} /> : <Play size={18} />}
          Test Model
        </button>
      </div>

      <div className="glass card flex-column" style={{ overflowY: 'auto' }}>
        <h3 className="micro-label">Results</h3>
        {result ? (
          <table className="results-table" style={{ fontSize: '0.8rem' }}>
            <thead><tr><th>Joint</th><th>X</th><th>Y</th></tr></thead>
            <tbody>
              {result.predictions.map((p, i) => (
                <tr key={i}><td>{p.joint}</td><td>{p.x.toFixed(0)}</td><td>{p.y.toFixed(0)}</td></tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="empty-state"><p className="text-secondary">No results yet</p></div>
        )}
      </div>
    </div>
  );
};

export default RunInference;
