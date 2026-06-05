import React, { useState } from 'react';
import { 
  Target, 
  FileJson, 
  Activity, 
  Terminal, 
  Play, 
  RefreshCw,
  AlertCircle,
  HelpCircle
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  BarChart,
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { evaluateModel, API_BASE_URL } from '../services/api';
import type { RunDetails } from '../pages/Overview';

interface ChartConfig {
  key: string;
  label: string;
  color: string;
  dash?: string;
}

interface HighlightConfig {
  key: string;
  label: string;
  color: 'primary' | 'lime' | 'pink' | 'coral';
  suffix?: string;
  multiplier?: number;
}

interface RunAnalysisProps {
  details: RunDetails;
  isActive?: boolean;
  trainingStatus?: {
    current_epoch: number;
    total_epochs: number;
    progress: number;
    loss_history: number[];
    val_loss_history: number[];
    adv_loss_history: number[];
    log_history: string[];
    current_metrics?: Record<string, number | string>;
    history_dict?: Record<string, Record<string, number | string | null>>;
    display_metadata?: {
      charts?: ChartConfig[];
      highlights?: HighlightConfig[];
      primary_metric?: string;
    };
  };
}

const RunAnalysis: React.FC<RunAnalysisProps> = ({ details, isActive, trainingStatus }) => {
  const [showLogs, setShowLogs] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [showLiveStats, setShowLiveStats] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  // Local override set by "Re-evaluate"; falls back to prop data
  const [localEvalOverride, setLocalEvalOverride] = useState<RunDetails['evaluation'] | null>(null);

  const evalResults = localEvalOverride ?? details.evaluation;

  // 1. Resolve Metadata
  const displayMetadata = trainingStatus?.display_metadata || details.display_metadata;
  
  // Heuristic fallbacks if metadata is missing (legacy support)
  const charts = displayMetadata?.charts || [
    { key: 'loss', label: 'Train Loss', color: '#6a5fc1' },
    { key: 'val_loss', label: 'Val Loss', color: '#c2ef4e', dash: '5 3' }
  ];

  const highlights = displayMetadata?.highlights || [
    { key: 'val_pck', label: 'VALIDATION PCK', color: 'lime', suffix: '%', multiplier: 100 },
    { key: 'loss', label: 'TRAIN LOSS', color: 'primary' }
  ];

  // Map backend color names to hex/vars
  const colorMap: Record<string, string> = {
    primary: '#6a5fc1',
    lime: '#c2ef4e',
    pink: '#fa7faa',
    coral: '#ffb287',
  };

  // Build chart data
  const chartData = (() => {
    // 1. Prefer active training status
    if (isActive && trainingStatus?.history_dict) {
      const historyDict = trainingStatus.history_dict;
      const epochs = Object.keys(historyDict).map(Number).sort((a, b) => a - b);
      
      return epochs.map(ep => {
        const metrics = historyDict[ep] || historyDict[String(ep)] || {};
        const entry: Record<string, number | null> = { epoch: ep };
        charts.forEach((c: ChartConfig) => {
          // Special handling for keys that might have multiple names
          let val = metrics[c.key];
          if (val === undefined && c.key === 'loss') val = metrics.loss_pose || metrics.train_loss;
          if (val === undefined && c.key === 'val_loss') val = metrics.val_loss_pose;
          
          entry[c.key] = typeof val === 'number' ? val : null;
        });
        return entry;
      });
    }

    // 2. Fallback to historical details.history
    if (details.history && details.history.length > 0) {
      return details.history.map((h: Record<string, number>, i: number) => {
        const entry: Record<string, number | null> = { epoch: h.epoch ?? (i + 1) };
        charts.forEach((c: ChartConfig) => {
          let val = h[c.key];
          if (val === undefined && c.key === 'loss') val = h.loss_pose || h.train_loss;
          if (val === undefined && c.key === 'val_loss') val = h.val_loss_pose;
          entry[c.key] = typeof val === 'number' ? val : null;
        });
        return entry;
      });
    }
    
    return [];
  })();

  const handleRunEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const results = await evaluateModel('val', undefined, details.id, true);
      setLocalEvalOverride(results);
    } catch (error) {
      console.error('Evaluation failed:', error);
      alert('Evaluation failed');
    } finally {
      setIsEvaluating(false);
    }
  };

  const hasPoseMetrics = typeof evalResults?.pck === 'number' && typeof evalResults?.mpjpe === 'number';

  const pckData = (evalResults?.per_joint_metrics || []).map((m: { name: string; pck: number }) => ({
    name: m.name.replace('_', ' '),
    pck: m.pck * 100
  }));

  const errorData = (evalResults?.per_joint_metrics || []).map((m: { name: string; error: number }) => ({
    name: m.name.replace('_', ' '),
    error: m.error
  }));

  return (
    <div className="run-analysis flex-column" style={{ gap: '24px', overflowY: 'auto', paddingRight: '10px' }}>
      
      {/* 1. Loss & Progress Section */}
        <div className="analysis-top-row" style={{ display: 'flex', gap: '24px', alignItems: 'stretch' }}>
          {/* Main Chart Card */}
          <div className="glass card" style={{ flex: 1, margin: 0, display: 'flex', flexDirection: 'column' }}>
            <div className="card-header">
              <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-lime)' }}>
                {isActive ? 'Live Training Performance' : 'Training History'}
              </h3>
              <Activity size={18} color="var(--accent-lime)" />
            </div>
            
            <div style={{ height: '320px', width: '100%', marginTop: '10px', position: 'relative' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.5)" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="rgba(255,255,255,0.5)" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-purple)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  {charts.map((c) => (
                    <Line 
                      key={c.key}
                      type="monotone" 
                      dataKey={c.key} 
                      stroke={colorMap[c.color] || c.color} 
                      strokeWidth={c.key === (displayMetadata?.primary_metric || 'loss') ? 2.5 : 1.5} 
                      dot={{ r: 2, fill: colorMap[c.color] || c.color, strokeWidth: 0 }} 
                      strokeDasharray={c.dash}
                      name={c.label} 
                      connectNulls
                      animationDuration={300}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {isActive && trainingStatus && (
              <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255,255,255,0.03)', borderTop: '1px solid var(--border-purple)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span className="micro-label">Progress: Epoch {trainingStatus.current_epoch} / {trainingStatus.total_epochs}</span>
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                    {trainingStatus.current_metrics?.speed && (
                      <span className="micro-label" style={{ color: 'var(--accent-primary)' }}>{trainingStatus.current_metrics.speed} it/s</span>
                    )}
                    {trainingStatus.current_metrics?.eta && (
                      <span className="micro-label" style={{ opacity: 0.6 }}>ETA: {trainingStatus.current_metrics.eta}</span>
                    )}
                    <span style={{ fontSize: '0.8rem', color: 'var(--accent-lime)' }}>{Math.round(trainingStatus.progress * 100)}%</span>
                  </div>
                </div>
                <div className="progress-bar-container" style={{ margin: 0, height: '6px' }}>
                  <div className="progress-bar" style={{ width: `${trainingStatus.progress * 100}%` }}></div>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar Metrics (Active Only) */}
          {isActive && (
            <div className="flex-column" style={{ width: '280px', gap: '16px' }}>
              {(() => {
                const metrics = trainingStatus?.current_metrics || {};
                const historyDict = trainingStatus?.history_dict || {};
                
                return highlights.map((h: HighlightConfig, i: number) => {
                  let val: number | string | null | undefined = metrics[h.key];
                  // Heuristic for PCK if not in current batch metrics
                  if (val === undefined && (h.key === 'val_pck' || h.key === 'pck')) {
                    const epochs = Object.keys(historyDict).map(Number).sort((a, b) => b - a);
                    for (const ep of epochs) {
                      const m = historyDict[ep] || historyDict[String(ep)];
                      if (m && (m.val_pck !== undefined || m.pck !== undefined)) {
                        val = m.val_pck ?? m.pck;
                        break;
                      }
                    }
                  }

                  const displayVal = val !== undefined && val !== null
                    ? (Number(val) * (h.multiplier || 1)).toFixed(h.key.includes('pck') ? 2 : 4) 
                    : '--';

                  return (
                    <div key={i} className="glass highlight-card" style={{ padding: '20px', borderRadius: '16px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <div className="micro-label" style={{ opacity: 0.6, fontSize: '0.65rem' }}>{h.label}</div>
                      <div style={{ fontSize: '2.2rem', fontWeight: 'bold', color: colorMap[h.color] || h.color, marginTop: '8px' }}>
                        {displayVal}{val !== undefined && val !== null ? h.suffix : ''}
                      </div>
                    </div>
                  );
                });
              })()}
            </div>
          )}
        </div>

      {/* 1.2 Live Statistics Grid (Collapsible) */}
      {isActive && trainingStatus?.current_metrics && (
        <div className="glass card" style={{ padding: '20px' }}>
          <button 
            className="flex-row" 
            style={{ width: '100%', background: 'none', border: 'none', justifyContent: 'space-between', color: 'inherit', padding: 0 }}
            onClick={() => setShowLiveStats(!showLiveStats)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Activity size={18} color="var(--accent-primary)" />
              <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-primary)', margin: 0 }}>Live Statistics</h3>
              <span className="micro-label" style={{ opacity: 0.5 }}>PER BATCH</span>
            </div>
            <span style={{ opacity: 0.5 }}>{showLiveStats ? 'Hide' : 'Show'}</span>
          </button>
          
          {showLiveStats && (
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', 
              gap: '12px',
              marginTop: '16px'
            }}>
              {Object.entries(trainingStatus.current_metrics)
                .filter(([key]) => !['loss', 'loss_pose', 'train_loss', 'adv_loss', 'cycle_loss', 'id_loss', 'd_loss', 'speed', 'eta', 'elapsed', 'val_pck', 'pck', 'sigma'].includes(key))
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([key, value]) => (
                  <div key={key} style={{ 
                    padding: '12px', 
                    borderRadius: '12px', 
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid var(--border-purple)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px'
                  }}>
                    <div className="micro-label" style={{ opacity: 0.5, fontSize: '0.6rem' }}>
                      {key.startsWith('batch_') ? `${key.replace('batch_', '').replace(/_/g, ' ')} (LIVE)` : key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: '1rem', fontWeight: 600, color: '#fff' }}>
                      {typeof value === 'number' ? value.toFixed(4) : value}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}

      {/* 2. Evaluation Section */}
      <div className="glass card">
        <div className="card-header">
          <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-primary)' }}>Quantitative Evaluation</h3>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              className="btn-tab" 
              onClick={handleRunEvaluation} 
              disabled={isEvaluating || isActive}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              {isEvaluating ? <RefreshCw size={14} className="spin" /> : <Play size={14} />}
              {evalResults ? 'Re-evaluate' : 'Run Evaluation'}
            </button>
            <Target size={18} color="var(--accent-primary)" />
          </div>
        </div>

        {evalResults ? (
          <div className="flex-column" style={{ gap: '24px', marginTop: '20px' }}>
            {hasPoseMetrics ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span className="micro-label text-secondary">Mean PCK@0.2</span>
                      <div className="tooltip-container">
                        <HelpCircle size={12} className="info-icon" />
                        <div className="tooltip-content">
                          Percentage of Correct Keypoints: A prediction is correct if it falls within 0.2 * Torso Diameter of the ground truth.
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-lime)' }}>
                      {evalResults.pck !== undefined ? (evalResults.pck * 100).toFixed(1) : 'N/A'}%
                    </div>
                  </div>
                  <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span className="micro-label text-secondary">Mean MPJPE</span>
                      <div className="tooltip-container">
                        <HelpCircle size={12} className="info-icon" />
                        <div className="tooltip-content">
                          Mean Per Joint Position Error: The average Euclidean distance (in pixels) between predicted and ground truth joint coordinates.
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-pink)' }}>
                      {evalResults.mpjpe !== undefined ? evalResults.mpjpe.toFixed(1) : 'N/A'}px
                    </div>
                  </div>
                  <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span className="micro-label text-secondary">Avg Val Loss</span>
                      <div className="tooltip-container">
                        <HelpCircle size={12} className="info-icon" />
                        <div className="tooltip-content">
                          The average Mean Squared Error (MSE) loss calculated over the validation set.
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 600, marginTop: '8px' }}>
                      {evalResults.loss?.toFixed(6) || 'N/A'}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', height: '300px' }}>
                  <div className="flex-column">
                    <span className="micro-label" style={{ marginBottom: '10px', opacity: 0.7 }}>PCK per Joint (%)</span>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={pckData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-purple)" horizontal={true} vertical={false} />
                        <XAxis type="number" domain={[0, 100]} hide />
                        <YAxis type="category" dataKey="name" stroke="var(--text-secondary)" fontSize={10} width={80} />
                        <Tooltip 
                          cursor={{ fill: 'rgba(255,255,255,0.05)' }} 
                          contentStyle={{ background: 'var(--bg-secondary)', border: 'none', borderRadius: '8px' }}
                          itemStyle={{ color: '#fff' }}
                          labelStyle={{ color: '#fff' }}
                        />
                        <Bar dataKey="pck" radius={[0, 4, 4, 0]}>
                          {pckData.map((entry: { name: string; pck: number }, index: number) => (
                            <Cell key={`cell-${index}`} fill={entry.pck > 80 ? 'var(--accent-lime)' : 'var(--accent-primary)'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex-column">
                    <span className="micro-label" style={{ marginBottom: '10px', opacity: 0.7 }}>MPJPE per Joint (px)</span>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={errorData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-purple)" vertical={false} />
                        <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={9} tick={{ angle: -45, textAnchor: 'end' }} height={60} interval={0} />
                        <YAxis stroke="var(--text-secondary)" fontSize={10} domain={['auto', 'auto']} />
                        <Tooltip 
                          cursor={{ fill: 'rgba(255,255,255,0.05)' }} 
                          contentStyle={{ background: 'var(--bg-secondary)', border: 'none', borderRadius: '8px' }}
                          itemStyle={{ color: '#fff' }}
                          labelStyle={{ color: '#fff' }}
                        />
                        <Bar dataKey="error" fill="var(--accent-pink)" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            ) : (
              <div className="glass" style={{ padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.01)' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-lime)' }} />
                    <span className="micro-label" style={{ opacity: 0.8, textTransform: 'uppercase' }}>Evaluation Successful</span>
                  </div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 600, color: '#fff' }}>
                    Domain Translation Model Details
                  </div>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>
                    This model (type: <span style={{ color: 'var(--accent-lime)' }}>{evalResults.model_type || 'domain translation'}</span>) has run evaluation successfully. Since it translates images between domains rather than predicting pose keypoints directly, no pose coordinate metrics (PCK/MPJPE) are produced.
                  </p>
                </div>
              </div>
            )}

            {evalResults.visual_audit && (
              <div className="flex-column" style={{ gap: '12px', marginTop: '12px' }}>
                <span className="micro-label text-secondary" style={{ textTransform: 'uppercase' }}>Visual Audit Output</span>
                <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', background: 'rgba(0,0,0,0.2)' }}>
                  <img 
                    src={
                      evalResults.visual_audit.startsWith('results/runs/')
                        ? `${API_BASE_URL}/static/runs/${evalResults.visual_audit.substring('results/runs/'.length)}`
                        : evalResults.visual_audit.startsWith('/static/runs/')
                        ? `${API_BASE_URL}${evalResults.visual_audit}`
                        : evalResults.visual_audit.startsWith('http')
                        ? evalResults.visual_audit
                        : `${API_BASE_URL}/${evalResults.visual_audit}`
                    } 
                    alt="Visual Audit"
                    style={{ width: '100%', height: 'auto', borderRadius: '8px', display: 'block', border: '1px solid rgba(255,255,255,0.05)' }} 
                  />
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '40px' }}>
            <AlertCircle size={32} opacity={0.5} />
            <p className="text-secondary">No evaluation data available for this run.</p>
          </div>
        )}
      </div>

      {/* 3. Collapsible Utils */}
      <div className="flex-column" style={{ gap: '12px' }}>
        <div className="glass card">
          <button 
            className="flex-row" 
            style={{ width: '100%', background: 'none', border: 'none', justifyContent: 'space-between', color: 'inherit' }}
            onClick={() => setShowLogs(!showLogs)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Terminal size={18} color="var(--accent-primary)" />
              <span className="text-uppercase micro-label">Process Logs</span>
            </div>
            <span style={{ opacity: 0.5 }}>{showLogs ? 'Hide' : 'Show'}</span>
          </button>
          {showLogs && (
            <div className="logs-viewer" style={{ marginTop: '16px', maxHeight: '300px', overflowY: 'auto', background: 'rgba(0,0,0,0.3)', padding: '12px', borderRadius: '8px', fontSize: '0.75rem' }}>
              {isActive ? trainingStatus?.log_history.map((log: string, i: number) => <div key={i}>{log}</div>) : <p className="text-secondary">Logs only available for active sessions.</p>}
            </div>
          )}
        </div>

        <div className="glass card">
          <button 
            className="flex-row" 
            style={{ width: '100%', background: 'none', border: 'none', justifyContent: 'space-between', color: 'inherit' }}
            onClick={() => setShowConfig(!showConfig)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <FileJson size={18} color="var(--accent-pink)" />
              <span className="text-uppercase micro-label">Run Configuration</span>
            </div>
            <span style={{ opacity: 0.5 }}>{showConfig ? 'Hide' : 'Show'}</span>
          </button>
          {showConfig && (
            <pre style={{ marginTop: '16px', fontSize: '0.75rem', background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '8px', overflow: 'auto', maxHeight: '400px' }}>
              {JSON.stringify(details.config, null, 2)}
            </pre>
          )}
        </div>
      </div>

    </div>
  );
};

export default RunAnalysis;
