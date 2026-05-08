import React, { useState, useEffect } from 'react';
import { 
  History, 
  Trash2, 
  Calendar, 
  Target, 
  Activity,
  FileJson,
  Package
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { getRuns, getRunDetails, deleteRun, evaluateModel } from '../services/api';

interface RunSummary {
  id: string;
  created_at: string;
  epochs?: number;
  final_loss?: number;
  final_val_loss?: number;
}

interface RunDetails {
  id: string;
  config?: Record<string, unknown>;
  history?: { epoch: number; train_loss: number; val_loss?: number }[];
  checkpoints?: { name: string; size_mb: number }[];
}

const RunsHistory: React.FC = () => {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunDetails | null>(null);
  const [evaluating, setEvaluating] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    let mounted = true;
    const loadRuns = async () => {
      try {
        const data = await getRuns();
        if (mounted) {
          setRuns(data.runs);
        }
      } catch (error) {
        console.error('Failed to fetch runs:', error);
      }
    };
    
    loadRuns();
    return () => {
      mounted = false;
    };
  }, [refreshTrigger]);

  const handleViewDetails = async (runId: string) => {
    try {
      const details = await getRunDetails(runId);
      setSelectedRun(details);
    } catch (error) {
      console.error('Failed to fetch run details:', error);
    }
  };

  const handleDeleteRun = async (e: React.MouseEvent, runId: string) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete run ${runId}?`)) return;
    
    try {
      await deleteRun(runId);
      if (selectedRun?.id === runId) setSelectedRun(null);
      setRefreshTrigger(prev => prev + 1);
    } catch (error) {
      console.error('Failed to delete run:', error);
    }
  };

  const handleEvaluateBest = async (runId: string) => {
    setEvaluating(runId);
    try {
      const result = await evaluateModel('val', 'best_model.pth', runId);
      alert(`Evaluation complete!\nMPJPE: ${result.mpjpe.toFixed(2)}\nPCK: ${result.pck.toFixed(2)}`);
    } catch (error) {
      console.error('Evaluation failed:', error);
      alert('Evaluation failed');
    } finally {
      setEvaluating(null);
    }
  };

  const chartData = selectedRun?.history?.map((entry) => ({
    epoch: entry.epoch,
    loss: entry.train_loss,
    val_loss: entry.val_loss
  })) || [];

  return (
    <div className="history-page">
      <div className="page-header">
        <h1 className="text-uppercase">Runs History</h1>
        <p className="text-secondary">Review past training sessions, models, and performance metrics.</p>
      </div>

      <div className="history-grid" style={{ display: 'grid', gridTemplateColumns: selectedRun ? '400px 1fr' : '1fr', gap: '24px', transition: 'all 0.3s ease' }}>
        <div className="runs-list-container flex-column">
          <div className="glass card">
            <div className="card-header">
              <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-primary)' }}>Training Runs</h3>
              <History size={18} color="var(--accent-primary)" />
            </div>
            
            <div className="runs-list" style={{ maxHeight: 'calc(100vh - 250px)', overflowY: 'auto' }}>
              {runs.length === 0 ? (
                <div style={{ padding: '40px', textAlign: 'center', opacity: 0.5 }}>
                  <p>No training runs found.</p>
                </div>
              ) : (
                runs.map(run => (
                  <div 
                    key={run.id} 
                    className={`run-item ${selectedRun?.id === run.id ? 'active' : ''}`}
                    onClick={() => handleViewDetails(run.id)}
                    style={{ 
                      padding: '16px', 
                      borderRadius: '8px', 
                      marginBottom: '12px', 
                      cursor: 'pointer',
                      background: selectedRun?.id === run.id ? 'rgba(168, 85, 247, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                      border: `1px solid ${selectedRun?.id === run.id ? 'var(--accent-primary)' : 'transparent'}`,
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                      <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{run.id}</span>
                      <button 
                        onClick={(e) => handleDeleteRun(e, run.id)}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-pink)', padding: '4px', cursor: 'pointer', opacity: 0.6 }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '12px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Calendar size={12} />
                        <span>{new Date(run.created_at).toLocaleDateString()}</span>
                      </div>
                      {run.epochs && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Activity size={12} />
                          <span>{run.epochs} epochs</span>
                        </div>
                      )}
                    </div>
                    
                    {run.final_loss !== undefined && (
                      <div style={{ marginTop: '8px', display: 'flex', gap: '12px' }}>
                        <div className="metric-tag">Loss: {run.final_loss.toFixed(4)}</div>
                        {run.final_val_loss !== undefined && (
                          <div className="metric-tag" style={{ background: 'rgba(132, 204, 22, 0.1)', color: 'var(--accent-lime)' }}>Val: {run.final_val_loss.toFixed(4)}</div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {selectedRun && (
          <div className="run-details-container flex-column">
            <div className="glass card detail-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: '1.5rem' }}>{selectedRun.id}</h2>
                  <p className="text-secondary" style={{ margin: '4px 0 0 0' }}>Detailed analysis and artifacts</p>
                </div>
                <div style={{ display: 'flex', gap: '12px' }}>
                   <button 
                    className="btn-lime btn-sm" 
                    onClick={() => handleEvaluateBest(selectedRun.id)}
                    disabled={evaluating === selectedRun.id}
                  >
                    <Target size={16} />
                    {evaluating === selectedRun.id ? 'Evaluating...' : 'Evaluate Best Model'}
                  </button>
                </div>
              </div>

              <div className="details-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '24px' }}>
                <div className="flex-column" style={{ gap: '24px' }}>
                  <div className="chart-section glass" style={{ padding: '20px', borderRadius: '12px', background: 'rgba(0,0,0,0.2)' }}>
                    <h3 className="micro-label text-uppercase" style={{ marginBottom: '16px' }}>Training & Validation Loss</h3>
                    <div style={{ height: '300px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-purple)" vertical={false} />
                          <XAxis dataKey="epoch" stroke="var(--text-secondary)" fontSize={12} tickLine={false} axisLine={false} />
                          <YAxis stroke="var(--text-secondary)" fontSize={12} tickLine={false} axisLine={false} />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'var(--bg-secondary)', 
                              borderColor: 'var(--border-purple)',
                              borderRadius: '8px'
                            }} 
                          />
                          <Line type="monotone" dataKey="loss" stroke="var(--accent-primary)" strokeWidth={2} dot={false} name="Train Loss" />
                          <Line type="monotone" dataKey="val_loss" stroke="var(--accent-lime)" strokeWidth={2} dot={false} name="Val Loss" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {selectedRun.config && (
                    <div className="config-section glass" style={{ padding: '20px', borderRadius: '12px', background: 'rgba(0,0,0,0.2)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                        <FileJson size={16} color="var(--accent-primary)" />
                        <h3 className="micro-label text-uppercase" style={{ margin: 0 }}>Configuration</h3>
                      </div>
                      <pre style={{ 
                        fontSize: '0.8rem', 
                        background: 'rgba(0,0,0,0.3)', 
                        padding: '16px', 
                        borderRadius: '8px', 
                        overflow: 'auto',
                        maxHeight: '200px'
                      }}>
                        {JSON.stringify(selectedRun.config, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>

                <div className="flex-column" style={{ gap: '24px' }}>
                  <div className="checkpoints-section glass" style={{ padding: '20px', borderRadius: '12px', background: 'rgba(0,0,0,0.2)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                      <Package size={16} color="var(--accent-lime)" />
                      <h3 className="micro-label text-uppercase" style={{ margin: 0 }}>Checkpoints</h3>
                    </div>
                    <div className="checkpoints-list" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {selectedRun.checkpoints?.map((ckpt, i) => (
                        <div key={i} style={{ 
                          padding: '10px 12px', 
                          background: 'rgba(255,255,255,0.05)', 
                          borderRadius: '6px',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          fontSize: '0.8rem'
                        }}>
                          <span style={{ color: ckpt.name === 'best_model.pth' ? 'var(--accent-lime)' : 'inherit', fontWeight: ckpt.name === 'best_model.pth' ? 600 : 400 }}>
                            {ckpt.name}
                          </span>
                          <span className="text-secondary" style={{ fontSize: '0.7rem' }}>{ckpt.size_mb.toFixed(1)} MB</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .history-grid {
          height: calc(100vh - 180px);
        }
        .run-item:hover {
          background: rgba(255, 255, 255, 0.06) !important;
        }
        .metric-tag {
          font-size: 0.7rem;
          padding: 2px 8px;
          border-radius: 4px;
          background: rgba(168, 85, 247, 0.1);
          color: var(--accent-primary);
          font-weight: 600;
        }
        .btn-sm {
          padding: 8px 16px;
          font-size: 0.8rem;
          height: auto;
        }
      `}</style>
    </div>
  );
};

export default RunsHistory;
