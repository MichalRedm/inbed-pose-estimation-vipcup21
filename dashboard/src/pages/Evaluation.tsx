import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Target, 
  AlertCircle,
  Play,
  Box,
  Layers,
  Search,
  Info
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { getModels, evaluateModel } from '../services/api';

interface Model {
  name: string;
  path: string;
  size_mb: number;
}

interface PerJointMetric {
  name: string;
  error: number;
  pck: number;
}

interface EvaluationResults {
  loss: number;
  mpjpe: number;
  pck: number;
  per_joint_metrics: PerJointMetric[];
}

const InfoTooltip = ({ text }: { text: string }) => (
  <div className="tooltip-container">
    <Info size={14} className="info-icon" />
    <div className="tooltip-content glass">
      {text}
    </div>
  </div>
);

const Evaluation: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedSplit, setSelectedSplit] = useState<string>('val');
  const [results, setResults] = useState<EvaluationResults | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchInitialData = async () => {
      setIsLoading(true);
      try {
        const data = await getModels();
        setModels(data.models);
        if (data.models.length > 0) {
          const latestModel = data.models[data.models.length - 1].name;
          setSelectedModel(latestModel);
          
          // Try to load cached results for the latest model
          try {
            const evalRes = await evaluateModel(selectedSplit, latestModel, false);
            setResults(evalRes);
          } catch {
            console.log('No cached results available yet');
          }
        }
      } catch (err) {
        console.error('Failed to fetch initial evaluation data:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchInitialData();
  }, [selectedSplit]);

  const handleEvaluate = async (force: boolean = true) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await evaluateModel(selectedSplit, selectedModel, force);
      setResults(data);
    } catch (err) {
      console.error('Evaluation failed:', err);
      setError('Evaluation failed. Please check the backend connection.');
    } finally {
      setIsLoading(false);
    }
  };

  const pckData = results?.per_joint_metrics.map(m => ({
    name: m.name.replace('_', ' '),
    pck: m.pck * 100
  })) || [];

  const errorData = results?.per_joint_metrics.map(m => ({
    name: m.name.replace('_', ' '),
    error: m.error
  })) || [];

  return (
    <div className="evaluation-page">
      <div className="page-header">
        <h1 className="text-uppercase">Model Evaluation</h1>
        <p className="text-secondary">Quantify performance on validation or test sets.</p>
      </div>

      <div className="evaluation-grid">
        <div className="controls-column flex-column">
          <div className="glass card controls-card">
            <div className="card-header">
              <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-primary)' }}>Configuration</h3>
              <Layers size={18} color="var(--accent-primary)" />
            </div>
            
            <div className="control-group">
              <label className="text-uppercase micro-label" style={{ display: 'block', marginBottom: '8px', opacity: 0.7 }}>Checkpoint</label>
              <div style={{ position: 'relative' }}>
                <select 
                  value={selectedModel} 
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="glass-input"
                  style={{ width: '100%', appearance: 'none', padding: '12px', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-purple)', borderRadius: '8px', color: 'var(--text-primary)' }}
                >
                  {models.map(m => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </select>
                <Box size={16} style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', opacity: 0.5, pointerEvents: 'none' }} />
              </div>
            </div>

            <div className="control-group" style={{ marginTop: '20px' }}>
              <label className="text-uppercase micro-label" style={{ display: 'block', marginBottom: '8px', opacity: 0.7 }}>Dataset Split</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  className={`btn-tab ${selectedSplit === 'train' ? 'active' : ''}`}
                  onClick={() => setSelectedSplit('train')}
                  style={{ flex: 1 }}
                >
                  Train
                </button>
                <button 
                  className={`btn-tab ${selectedSplit === 'val' ? 'active' : ''}`}
                  onClick={() => setSelectedSplit('val')}
                  style={{ flex: 1 }}
                >
                  Validation
                </button>
              </div>
            </div>

            <button 
              className="btn-primary" 
              onClick={() => handleEvaluate(true)} 
              disabled={isLoading || !selectedModel}
              style={{ width: '100%', marginTop: '24px' }}
            >
              {isLoading ? (
                <>
                  <RefreshCw size={18} className="spin" />
                  Evaluating...
                </>
              ) : (
                <>
                  <Play size={18} fill="currentColor" />
                  Run Evaluation
                </>
              )}
            </button>

            {error && (
              <div className="error-message" style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-pink)', fontSize: '0.85rem' }}>
                <AlertCircle size={16} />
                {error}
              </div>
            )}
          </div>

          {results && (
            <div className="glass card summary-card">
              <h3 className="text-uppercase micro-label" style={{ marginBottom: '20px' }}>Global Metrics</h3>
              
              <div className="metric-row" style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
                <div className="metric-item" style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span className="text-secondary micro-label">MPJPE</span>
                    <InfoTooltip text="Mean Per Joint Position Error. Average Euclidean distance between predicted and ground truth joints (in pixels)." />
                  </div>
                  <div className="metric-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{results.mpjpe.toFixed(2)} <span className="unit" style={{ fontSize: '0.8rem', opacity: 0.5 }}>px</span></div>
                </div>
                <div className="metric-item" style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span className="text-secondary micro-label">PCK @ 15px</span>
                    <InfoTooltip text="Percentage of Correct Keypoints. The percentage of joints where the error is below the 15px threshold." />
                  </div>
                  <div className="metric-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{(results.pck * 100).toFixed(1)} <span className="unit" style={{ fontSize: '0.8rem', opacity: 0.5 }}>%</span></div>
                </div>
              </div>
              
              <div className="metric-item">
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span className="text-secondary micro-label">Avg Loss</span>
                  <InfoTooltip text="Average Mean Squared Error (MSE) of the predicted heatmaps compared to the ground truth." />
                </div>
                <div className="metric-value" style={{ fontSize: '1.1rem' }}>{results.loss.toFixed(6)}</div>
              </div>
            </div>
          )}
        </div>

        <div className="results-column">
          {results ? (
            <div className="flex-column">
              <div className="glass card chart-container">
                <div className="card-header">
                  <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-lime)' }}>PCK per Joint (%)</h3>
                  <Target size={18} color="var(--accent-lime)" />
                </div>
                <div className="chart-wrapper" style={{ height: '350px', marginTop: '20px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={pckData} layout="vertical" margin={{ left: 10, right: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-purple)" horizontal={true} vertical={false} />
                      <XAxis type="number" domain={[0, 100]} stroke="var(--text-secondary)" fontSize={11} />
                      <YAxis type="category" dataKey="name" stroke="var(--text-secondary)" fontSize={11} width={100} interval={0} />
                      <Tooltip 
                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                        contentStyle={{ 
                          backgroundColor: 'rgba(15, 12, 28, 0.95)',
                          border: '1px solid var(--border-purple)',
                          borderRadius: '8px',
                          boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                        }}
                        itemStyle={{ color: 'var(--accent-lime)', fontSize: '0.9rem' }}
                        labelStyle={{ color: 'var(--text-primary)', fontWeight: 'bold', marginBottom: '4px' }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(value: any) => [`${Number(value).toFixed(2)} %`, 'PCK']}
                      />
                      <Bar dataKey="pck" radius={[0, 4, 4, 0]}>
                        {pckData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.pck > 80 ? 'var(--accent-lime)' : entry.pck > 50 ? 'var(--accent-primary)' : 'var(--accent-pink)'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="glass card chart-container">
                <div className="card-header">
                  <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-pink)' }}>MPJPE per Joint (px)</h3>
                  <BarChart3 size={18} color="var(--accent-pink)" />
                </div>
                <div className="chart-wrapper" style={{ height: '350px', marginTop: '20px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={errorData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-purple)" vertical={false} />
                      <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={10} tick={{ angle: -45, textAnchor: 'end' }} height={80} interval={0} />
                      <YAxis stroke="var(--text-secondary)" fontSize={11} />
                      <Tooltip 
                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                        contentStyle={{ 
                          backgroundColor: 'rgba(15, 12, 28, 0.95)',
                          border: '1px solid var(--border-purple)',
                          borderRadius: '8px',
                          boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                        }}
                        itemStyle={{ color: 'var(--accent-pink)', fontSize: '0.9rem' }}
                        labelStyle={{ color: 'var(--text-primary)', fontWeight: 'bold', marginBottom: '4px' }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(value: any) => [`${Number(value).toFixed(2)} px`, 'Error']}
                      />
                      <Bar dataKey="error" fill="var(--accent-pink)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass card empty-state" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
              <div className="empty-content" style={{ textAlign: 'center', opacity: 0.5 }}>
                <Search size={64} style={{ marginBottom: '20px' }} />
                <h3 className="text-uppercase">No Evaluation Results</h3>
                <p>Select a model and run evaluation to see detailed performance statistics.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Add missing icon for loading
const RefreshCw = ({ size, className }: { size: number, className: string }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={size} height={size} 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
  >
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
    <path d="M21 3v5h-5"></path>
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
    <path d="M8 16H3v5"></path>
  </svg>
);

export default Evaluation;
