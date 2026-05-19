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
import { evaluateModel } from '../services/api';
import type { RunDetails } from '../pages/Overview';

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
    history_dict?: Record<string, Record<string, number>>;
  };
}

interface HistoryMetrics {
  val_pck?: number;
  pck?: number;
  val_loss?: number;
  loss?: number;
  [key: string]: number | undefined;
}

const RunAnalysis: React.FC<RunAnalysisProps> = ({ details, isActive, trainingStatus }) => {
  const [showLogs, setShowLogs] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [showLiveStats, setShowLiveStats] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  // Local override set by "Re-evaluate"; falls back to prop data
  const [localEvalOverride, setLocalEvalOverride] = useState<RunDetails['evaluation'] | null>(null);

  // Note: All local state (showLogs, localEvalOverride, etc.) is naturally reset 
  // when the 'details.id' changes because the parent renders this component with a 'key={details.id}'.

  const evalResults = localEvalOverride ?? details.evaluation;

  // Build chart data - Unify active and historical logic
  const chartData = (() => {
    // 1. Prefer active training status if it's currently running
    if (isActive && trainingStatus?.loss_history && trainingStatus.loss_history.length > 0) {
      const history = trainingStatus.loss_history;
      const historyDict = trainingStatus.history_dict || {};
      const total = trainingStatus.total_epochs || 30;
      
      return history.map((loss, i) => {
        const ep = i + 1;
        // API dict keys become strings in JSON
        const metrics = (historyDict[ep] || historyDict[String(ep)]) as HistoryMetrics | undefined || {};
        return {
          epoch: ep,
          loss: loss ?? null,
          val_loss: (metrics.val_loss ?? metrics.val_loss_pose) ?? null,
          adv: trainingStatus.adv_loss_history?.[i] ?? null,
        };
      }).filter(d => d.epoch <= total);
    }

    // 2. Fallback to historical data (most complete for finished runs)
    if (details.history && details.history.length > 0) {
      return details.history.map((h: Record<string, number>, i) => ({
        epoch: h.epoch ?? (i + 1),
        loss: h.loss ?? h.loss_pose ?? h.train_loss ?? null,
        val_loss: h.val_loss ?? h.val_loss_pose ?? null,
        adv: h.adv_loss ?? null,
      }));
    }
    
    return [];
  })();

  const hasAdvHistory = isActive && !!trainingStatus?.adv_loss_history;

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
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#c2ef4e" 
                    strokeWidth={2.5} 
                    dot={{ r: 3, fill: '#c2ef4e', strokeWidth: 0 }} 
                    activeDot={{ r: 5 }} 
                    animationDuration={300} 
                    name="Train Loss" 
                    connectNulls
                  />
                  <Line 
                    type="monotone" 
                    dataKey="val_loss" 
                    stroke="#6a5fc1" 
                    strokeWidth={2} 
                    dot={{ r: 2, fill: '#6a5fc1', strokeWidth: 0 }} 
                    strokeDasharray="5 3" 
                    name="Val Loss" 
                    connectNulls
                  />
                  {hasAdvHistory && (
                    <Line 
                      type="monotone" 
                      dataKey="adv" 
                      stroke="#fa7faa" 
                      strokeWidth={1.5} 
                      dot={{ r: 2, fill: '#fa7faa', strokeWidth: 0 }} 
                      strokeDasharray="4 4" 
                      name="Adv Loss" 
                      connectNulls
                    />
                  )}
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
              <div className="glass highlight-card glow-lime" style={{ padding: '20px', borderRadius: '16px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div className="micro-label" style={{ opacity: 0.6, fontSize: '0.65rem' }}>VALIDATION PCK</div>
                <div style={{ fontSize: '2.2rem', fontWeight: 'bold', color: 'var(--accent-lime)', marginTop: '8px' }}>
                  {(() => {
                    const metrics = trainingStatus?.current_metrics || {};
                    // Try current metrics first (if just finished)
                    let pck: number | string | undefined = metrics.val_pck ?? metrics.pck;
                    
                    if (!pck && trainingStatus) {
                      // Scan history backwards for latest available PCK
                      const historyDict = trainingStatus.history_dict || {};
                      const epochs = Object.keys(historyDict).map(Number).sort((a, b) => b - a);
                      for (const ep of epochs) {
                        const m = (historyDict[ep] || historyDict[String(ep)]) as HistoryMetrics | undefined;
                        if (m && (m.val_pck !== undefined || m.pck !== undefined)) {
                          pck = m.val_pck ?? m.pck;
                          break;
                        }
                      }
                    }
                    return pck ? `${(Number(pck) * 100).toFixed(2)}%` : '--';
                  })()}
                </div>
              </div>
              
              <div className="glass highlight-card glow-purple" style={{ padding: '20px', borderRadius: '16px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div className="micro-label" style={{ opacity: 0.6, fontSize: '0.65rem' }}>LAST EPOCH LOSS</div>
                <div style={{ fontSize: '2.2rem', fontWeight: 'bold', color: 'var(--accent-primary)', marginTop: '8px' }}>
                  {trainingStatus ? (
                    (Number(trainingStatus.current_metrics?.loss) || Number(trainingStatus.current_metrics?.loss_pose) || 0).toFixed(4)
                  ) : '--'}
                </div>
              </div>

              <div className="glass highlight-card glow-pink" style={{ padding: '20px', borderRadius: '16px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div className="micro-label" style={{ opacity: 0.6, fontSize: '0.65rem' }}>SIGMA</div>
                <div style={{ fontSize: '2.2rem', fontWeight: 'bold', color: 'var(--accent-pink)', marginTop: '8px' }}>
                  {trainingStatus ? (
                    (Number(trainingStatus.current_metrics?.sigma) || 2.0).toFixed(3)
                  ) : '--'}
                </div>
              </div>
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
                .filter(([key]) => !['loss', 'train_loss', 'adv_loss', 'speed', 'eta', 'elapsed', 'val_pck', 'sigma'].includes(key))
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
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-lime)' }}>{(evalResults.pck * 100).toFixed(1)}%</div>
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
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-pink)' }}>{evalResults.mpjpe.toFixed(1)}px</div>
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
                <div style={{ fontSize: '1.2rem', fontWeight: 600, marginTop: '8px' }}>{evalResults.loss?.toFixed(6) || 'N/A'}</div>
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
