import React, { useState, useEffect } from 'react';
import { 
  Target, 
  FileJson, 
  BarChart3, 
  Activity, 
  Terminal, 
  Play, 
  RefreshCw,
  AlertCircle
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

interface RunAnalysisProps {
  details: {
    id: string;
    config?: any;
    evaluation?: any;
    history?: any[];
  };
  isActive?: boolean;
  trainingStatus?: any;
}

const RunAnalysis: React.FC<RunAnalysisProps> = ({ details, isActive, trainingStatus }) => {
  const [showLogs, setShowLogs] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  // Local override set by "Re-evaluate"; falls back to prop data
  const [localEvalOverride, setLocalEvalOverride] = useState<any>(null);

  // Reset the override when run changes (key prop also handles this but this is belt-and-suspenders)
  useEffect(() => {
    setLocalEvalOverride(null);
    setShowLogs(false);
    setShowConfig(false);
  }, [details.id]);

  const evalResults = localEvalOverride ?? details.evaluation;

  // Build chart data from history - API returns objects with named fields
  const chartData = (() => {
    if (isActive && trainingStatus?.loss_history) {
      return trainingStatus.loss_history.map((loss: number, i: number) => ({
        epoch: i + 1,
        loss,
        val_loss: trainingStatus.val_loss_history?.[i] ?? null,
        adv: trainingStatus.adv_loss_history?.[i] ?? null,
      }));
    }
    // History entries are objects: {epoch, loss, val_loss, val_pck, loss_pose, ...}
    return (details.history || []).map((h: any) => ({
      epoch: typeof h.epoch === 'number' ? h.epoch : (details.history!.indexOf(h) + 1),
      loss: typeof h.loss === 'number' ? h.loss : (typeof h.loss_pose === 'number' ? h.loss_pose : null),
      val_loss: typeof h.val_loss === 'number' ? h.val_loss : (typeof h.val_loss_pose === 'number' ? h.val_loss_pose : null),
      val_pck: typeof h.val_pck === 'number' ? h.val_pck : null,
    }));
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

  const pckData = (evalResults?.per_joint_metrics || []).map((m: any) => ({
    name: m.name.replace('_', ' '),
    pck: m.pck * 100
  }));

  const errorData = (evalResults?.per_joint_metrics || []).map((m: any) => ({
    name: m.name.replace('_', ' '),
    error: m.error
  }));

  return (
    <div className="run-analysis flex-column" style={{ gap: '24px', overflowY: 'auto', paddingRight: '10px' }}>
      
      {/* 1. Loss & Progress Section */}
      <div className="glass card">
        <div className="card-header">
          <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-lime)' }}>
            {isActive ? 'Live Training Performance' : 'Training History'}
          </h3>
          <Activity size={18} color="var(--accent-lime)" />
        </div>
        
        <div style={{ height: '300px', marginTop: '10px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-purple)" vertical={false} />
              <XAxis dataKey="epoch" stroke="var(--text-secondary)" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--text-secondary)" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-purple)', borderRadius: '8px' }} />
              <Line type="monotone" dataKey="loss" stroke="var(--accent-lime)" strokeWidth={2.5} dot={false} animationDuration={300} name="Train Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="var(--accent-primary)" strokeWidth={2} dot={false} strokeDasharray="5 3" name="Val Loss" />
              {hasAdvHistory && <Line type="monotone" dataKey="adv" stroke="var(--accent-pink)" strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="Adv Loss" />}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {isActive && trainingStatus && (
          <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
             <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span className="micro-label">Progress: Epoch {trainingStatus.current_epoch} / {trainingStatus.total_epochs}</span>
              <span style={{ fontSize: '0.8rem', color: 'var(--accent-lime)' }}>{Math.round(trainingStatus.progress * 100)}%</span>
            </div>
            <div className="progress-bar-container" style={{ margin: 0, height: '6px' }}>
              <div className="progress-bar" style={{ width: `${trainingStatus.progress * 100}%` }}></div>
            </div>
          </div>
        )}
      </div>

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
                <span className="micro-label text-secondary">Mean PCK@0.5</span>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-lime)' }}>{(evalResults.pck * 100).toFixed(1)}%</div>
              </div>
              <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <span className="micro-label text-secondary">Mean MPJPE</span>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-pink)' }}>{evalResults.mpjpe.toFixed(1)}px</div>
              </div>
              <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <span className="micro-label text-secondary">Avg Val Loss</span>
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
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ background: 'var(--bg-secondary)', border: 'none', borderRadius: '8px' }} />
                    <Bar dataKey="pck" radius={[0, 4, 4, 0]}>
                      {pckData.map((entry: any, index: number) => (
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
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ background: 'var(--bg-secondary)', border: 'none', borderRadius: '8px' }} />
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
