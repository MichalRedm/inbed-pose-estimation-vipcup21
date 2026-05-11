import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Square, 
  Activity,
  RefreshCw,
  Server
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
import { 
  getTrainingStatus, 
  startTraining, 
  stopTraining,
  getTrainingConfig,
  saveTrainingConfig
} from '../services/api';

interface TrainingConfig {
  lr: number;
  epochs: number;
  batch_size: number;
  remote: boolean;
  augmentation: {
    enabled: boolean;
    occlusion_prob: number;
    flip_prob: number;
    rotation_range: [number, number];
    scaling_range: [number, number];
  };
  uda: boolean;
  lambda_adv: number;
  anatomical: boolean;
  lambda_anatomical: number;
}

interface TrainingStatus {
  is_running: boolean;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  loss_history: number[];
  adv_loss_history: number[];
  log_history: string[];
  status_message: string;
  current_metrics?: Record<string, number>;
}

const Training: React.FC = () => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const logsContainerRef = React.useRef<HTMLDivElement>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [config, setConfig] = useState<TrainingConfig>({
    lr: 0.0001,
    epochs: 30,
    batch_size: 16,
    remote: false,
    augmentation: {
      enabled: false,
      occlusion_prob: 0.5,
      flip_prob: 0.5,
      rotation_range: [-30, 30],
      scaling_range: [0.8, 1.2]
    },
    uda: false,
    lambda_adv: 0.1,
    anatomical: false,
    lambda_anatomical: 0.01
  });

  // Load initial config
  useEffect(() => {
    const initialize = async () => {
      try {
        const savedConfig = await getTrainingConfig();
        if (savedConfig) {
          setConfig(prev => {
            const next = { ...prev };
            if (savedConfig.lr !== undefined) next.lr = savedConfig.lr;
            if (savedConfig.epochs !== undefined) next.epochs = savedConfig.epochs;
            if (savedConfig.batch_size !== undefined) next.batch_size = savedConfig.batch_size;
            if (savedConfig.remote !== undefined) next.remote = savedConfig.remote;
            if (savedConfig.uda !== undefined) next.uda = savedConfig.uda;
            if (savedConfig.lambda_adv !== undefined) next.lambda_adv = savedConfig.lambda_adv;
            if (savedConfig.lambda_anatomical !== undefined) {
              next.lambda_anatomical = savedConfig.lambda_anatomical;
              next.anatomical = savedConfig.lambda_anatomical > 0;
            }
            
            if (savedConfig.augmentation) {
              next.augmentation = {
                ...prev.augmentation,
                ...savedConfig.augmentation
              };
            }
            return next;
          });
        }
      } catch (error) {
        console.error('Failed to load training config:', error);
      }
    };
    initialize();
  }, []);

  // Auto-save config changes with debounce
  useEffect(() => {
    // Skip saving on initial load or if already idle
    if (saveStatus === 'idle') return;

    const saveTimer = setTimeout(async () => {
      try {
        await saveTrainingConfig(config as unknown as Record<string, unknown>);
        setSaveStatus('saved');
        setTimeout(() => setSaveStatus('idle'), 2000);
      } catch (error) {
        console.error('Auto-save failed:', error);
        setSaveStatus('error');
      }
    }, 1000);

    return () => clearTimeout(saveTimer);
  }, [config, saveStatus]);

  // Wrap setConfig to trigger the "saving" state immediately for UI feedback
  const updateConfig = (updater: (prev: TrainingConfig) => TrainingConfig) => {
    setSaveStatus('saving');
    setConfig(updater);
  };

  const fetchStatus = React.useCallback(async () => {
    try {
      const data = await getTrainingStatus();
      setStatus(data);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  }, []);

  useEffect(() => {
    const load = async () => {
      await fetchStatus();
    };
    load();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Auto-scroll logs only if user is already at the bottom
  useEffect(() => {
    const container = logsContainerRef.current;
    if (container) {
      const threshold = 100; // px
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + threshold;
      
      if (isAtBottom) {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'auto'
        });
      }
    }
  }, [status?.log_history]);

  const handleStart = async () => {
    try {
      // Persist the current configuration to the backend first
      await saveTrainingConfig({
        lr: config.lr,
        epochs: config.epochs,
        batch_size: config.batch_size,
        remote: config.remote,
        augmentation: config.augmentation,
        uda: config.uda,
        lambda_adv: config.lambda_adv,
        lambda_anatomical: config.anatomical ? config.lambda_anatomical : 0
      });

      // Start training with the current configuration
      await startTraining({
        augmentation: config.augmentation,
        uda: config.uda,
        lambda_adv: config.lambda_adv,
        lambda_anatomical: config.anatomical ? config.lambda_anatomical : 0,
        remote: config.remote
      });
      fetchStatus();
    } catch (error) {
      console.error('Failed to start training:', error);
      alert('Failed to start training');
    }
  };


  const handleStop = async () => {
    try {
      await stopTraining();
      fetchStatus();
    } catch {
      alert('Failed to stop training');
    }
  };

  const chartData = status?.loss_history.map((loss, index) => ({
    epoch: index + 1,
    loss: loss,
    adv_loss: status.adv_loss_history ? status.adv_loss_history[index] : 0
  })) || [];

  return (
    <div className="training-page">
      <div className="page-header">
        <h1 className="text-uppercase">Training Monitor</h1>
        <p className="text-secondary">Track and control model training sessions.</p>
      </div>

      <div className="training-grid">
        <div className="main-stats flex-column">
          <div className="glass card chart-container">
            <div className="card-header">
              <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-lime)' }}>Loss History</h3>
              <Activity size={18} color="var(--accent-lime)" />
            </div>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-purple)" vertical={false} />
                  <XAxis 
                    dataKey="epoch" 
                    stroke="var(--text-secondary)" 
                    fontSize={12} 
                    tickLine={false} 
                    axisLine={false} 
                  />
                  <YAxis 
                    stroke="var(--text-secondary)" 
                    fontSize={12} 
                    tickLine={false} 
                    axisLine={false} 
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'var(--bg-secondary)', 
                      borderColor: 'var(--border-purple)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="var(--accent-lime)" 
                    strokeWidth={3} 
                    dot={{ r: 4, fill: 'var(--accent-lime)' }}
                    activeDot={{ r: 6, stroke: 'white', strokeWidth: 2 }}
                    name="Pose Loss"
                  />
                  {config.uda && (
                    <Line 
                      type="monotone" 
                      dataKey="adv_loss" 
                      stroke="var(--accent-pink)" 
                      strokeWidth={2} 
                      dot={false}
                      name="Adv Loss"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {status?.is_running && (
            <div className="metrics-highlight-row" style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '20px', 
              marginBottom: '20px' 
            }}>
              <div className="glass card highlight-card" style={{ borderLeft: '4px solid var(--accent-lime)' }}>
                <div className="micro-label" style={{ opacity: 0.6 }}>VALIDATION PCK</div>
                <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--accent-lime)', marginTop: '8px' }}>
                  {status.current_metrics?.val_pck ? `${status.current_metrics.val_pck.toFixed(2)}%` : '--'}
                </div>
                <div style={{ fontSize: '0.7rem', opacity: 0.5, marginTop: '4px' }}>Latest from validation split</div>
              </div>
              <div className="glass card highlight-card" style={{ borderLeft: '4px solid var(--accent-primary)' }}>
                <div className="micro-label" style={{ opacity: 0.6 }}>CURRENT LOSS</div>
                <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--accent-primary)', marginTop: '8px' }}>
                  {status.current_metrics?.loss ? status.current_metrics.loss.toFixed(4) : '--'}
                </div>
                <div style={{ fontSize: '0.7rem', opacity: 0.5, marginTop: '4px' }}>Last batch training loss</div>
              </div>
              <div className="glass card highlight-card" style={{ borderLeft: '4px solid var(--accent-pink)' }}>
                <div className="micro-label" style={{ opacity: 0.6 }}>ADAPTIVE SIGMA</div>
                <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--accent-pink)', marginTop: '8px' }}>
                  {status.current_metrics?.sigma ? status.current_metrics.sigma.toFixed(3) : '--'}
                </div>
                <div style={{ fontSize: '0.7rem', opacity: 0.5, marginTop: '4px' }}>Heatmap Gaussian spread</div>
              </div>
            </div>
          )}

          <div className="glass card progress-card">
            <div className="card-header">
              <h3 className="text-uppercase micro-label">Overall Progress</h3>
              <span className="text-secondary">{status?.current_epoch || 0} / {status?.total_epochs || 0} Epochs</span>
            </div>
            <div className="progress-bar-container">
              <div className="progress-bar" style={{ width: `${(status?.progress || 0) * 100}%` }}></div>
            </div>
            <div className="status-footer" style={{ justifyContent: 'space-between', padding: '4px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span className="text-secondary" style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                  {status?.status_message || 'Idle'}
                </span>
                {status?.is_running && <RefreshCw size={14} className="spin" style={{ marginLeft: '12px', color: 'var(--accent-lime)' }} />}
              </div>
              {status?.is_running && (
                <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                  {status.current_metrics?.speed && (
                    <span className="micro-label" style={{ color: 'var(--accent-primary)', background: 'rgba(56, 189, 248, 0.1)', padding: '2px 6px', borderRadius: '4px' }}>
                      {status.current_metrics.speed} it/s
                    </span>
                  )}
                  {status.current_metrics?.eta && (
                    <span className="micro-label" style={{ opacity: 0.8 }}>
                      ETA: {status.current_metrics.eta}
                    </span>
                  )}
                  <span className="micro-label" style={{ color: 'var(--accent-primary)', fontWeight: 'bold' }}>LIVE</span>
                </div>
              )}
            </div>
          </div>

          {status?.current_metrics && Object.keys(status.current_metrics).length > 0 && (
            <div className="glass card metrics-card" style={{ marginTop: '20px' }}>
              <div className="card-header">
                <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-primary)' }}>Live Statistics</h3>
                <span className="micro-label" style={{ opacity: 0.5 }}>UPDATED PER BATCH</span>
              </div>
              <div className="metrics-grid" style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', 
                gap: '12px',
                marginTop: '16px'
              }}>
                {Object.entries(status.current_metrics)
                  .filter(([key]) => !['loss', 'train_loss', 'adv_loss'].includes(key)) // Primary losses already in chart
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([key, value]) => (
                    <div key={key} className="metric-item" style={{ 
                      background: 'rgba(255,255,255,0.03)', 
                      padding: '10px', 
                      borderRadius: '8px',
                      borderLeft: `2px solid ${key.includes('sigma') ? 'var(--accent-pink)' : key.includes('pck') ? 'var(--accent-lime)' : 'var(--accent-primary)'}`
                    }}>
                      <div className="text-uppercase" style={{ fontSize: '0.6rem', opacity: 0.6, marginBottom: '4px' }}>
                        {key.replace(/_/g, ' ')}
                      </div>
                      <div style={{ fontSize: '1rem', fontWeight: 'bold', fontFamily: 'monospace' }}>
                        {typeof value === 'number' ? (value > 1 ? value.toFixed(2) : value.toFixed(4)) : value}
                        {key.includes('pck') && '%'}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>

        <div className="side-controls flex-column">
          <div className="glass card controls-card">
            <h3 className="text-uppercase micro-label" style={{ marginBottom: '20px' }}>Controls</h3>
            <div className="control-group" style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', borderRadius: '8px', background: 'rgba(255,255,255,0.05)', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Server size={18} color="var(--accent-primary)" />
                  <span style={{ fontSize: '0.85rem' }}>Remote Training</span>
                </div>
                <label className="switch" style={{ position: 'relative', display: 'inline-block', width: '40px', height: '20px' }}>
                  <input 
                    type="checkbox" 
                    checked={config.remote}
                    onChange={(e) => updateConfig(prev => ({...prev, remote: e.target.checked}))}
                    style={{ opacity: 0, width: 0, height: 0 }}
                  />
                  <span className="slider" style={{ 
                    position: 'absolute', cursor: 'pointer', top: 0, left: 0, right: 0, bottom: 0, 
                    backgroundColor: config.remote ? 'var(--accent-primary)' : '#555', 
                    transition: '.4s', borderRadius: '20px' 
                  }}>
                    <span style={{ 
                      position: 'absolute', height: '16px', width: '16px', left: config.remote ? '22px' : '2px', 
                      bottom: '2px', backgroundColor: 'white', transition: '.4s', borderRadius: '50%' 
                    }}></span>
                  </span>
                </label>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', borderRadius: '8px', background: 'rgba(255,255,255,0.05)', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Activity size={18} color="var(--accent-pink)" />
                  <span style={{ fontSize: '0.85rem' }}>Adversarial Alignment (UDA)</span>
                </div>
                <label className="switch" style={{ position: 'relative', display: 'inline-block', width: '40px', height: '20px' }}>
                  <input 
                    type="checkbox" 
                    checked={config.uda}
                    onChange={(e) => updateConfig(prev => ({...prev, uda: e.target.checked}))}
                    style={{ opacity: 0, width: 0, height: 0 }}
                  />
                  <span className="slider" style={{ 
                    position: 'absolute', cursor: 'pointer', top: 0, left: 0, right: 0, bottom: 0, 
                    backgroundColor: config.uda ? 'var(--accent-pink)' : '#555', 
                    transition: '.4s', borderRadius: '20px' 
                  }}>
                    <span style={{ 
                      position: 'absolute', height: '16px', width: '16px', left: config.uda ? '22px' : '2px', 
                      bottom: '2px', backgroundColor: 'white', transition: '.4s', borderRadius: '50%' 
                    }}></span>
                  </span>
                </label>
              </div>

              {config.uda && (
                <div className="input-field" style={{ marginBottom: '16px', marginLeft: '12px' }}>
                  <label>Lambda Adv</label>
                  <input 
                    type="number" 
                    value={config.lambda_adv} 
                    onChange={(e) => updateConfig(prev => ({...prev, lambda_adv: parseFloat(e.target.value)}))}
                    step="0.05" min="0"
                  />
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', borderRadius: '8px', background: 'rgba(255,255,255,0.05)', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Activity size={18} color="var(--accent-lime)" />
                  <span style={{ fontSize: '0.85rem' }}>Anatomical Constraints</span>
                </div>
                <label className="switch" style={{ position: 'relative', display: 'inline-block', width: '40px', height: '20px' }}>
                  <input 
                    type="checkbox" 
                    checked={config.anatomical}
                    onChange={(e) => updateConfig(prev => ({...prev, anatomical: e.target.checked}))}
                    style={{ opacity: 0, width: 0, height: 0 }}
                  />
                  <span className="slider" style={{ 
                    position: 'absolute', cursor: 'pointer', top: 0, left: 0, right: 0, bottom: 0, 
                    backgroundColor: config.anatomical ? 'var(--accent-lime)' : '#555', 
                    transition: '.4s', borderRadius: '20px' 
                  }}>
                    <span style={{ 
                      position: 'absolute', height: '16px', width: '16px', left: config.anatomical ? '22px' : '2px', 
                      bottom: '2px', backgroundColor: 'white', transition: '.4s', borderRadius: '50%' 
                    }}></span>
                  </span>
                </label>
              </div>

              {config.anatomical && (
                <div className="input-field" style={{ marginBottom: '16px', marginLeft: '12px' }}>
                  <label>Lambda Anatomical</label>
                  <input 
                    type="number" 
                    value={config.lambda_anatomical} 
                    onChange={(e) => updateConfig(prev => ({...prev, lambda_anatomical: parseFloat(e.target.value)}))}
                    step="0.01" min="0"
                  />
                </div>
              )}

              {status?.is_running ? (

                <button className="btn-primary" onClick={handleStop} style={{ width: '100%', background: 'var(--accent-pink)', borderColor: '#8a4d4d' }}>
                  <Square size={18} fill="currentColor" />
                  Stop Training
                </button>
              ) : (
                <button className="btn-lime" onClick={handleStart} style={{ width: '100%' }}>
                  <Play size={18} fill="currentColor" />
                  Start Training
                </button>
              )}
            </div>

            <div className="config-section" style={{ marginTop: '32px', position: 'relative' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <h4 className="text-uppercase micro-label" style={{ opacity: 0.7, margin: 0 }}>Hyperparameters</h4>
                {saveStatus !== 'idle' && (
                  <span style={{ 
                    fontSize: '0.65rem', 
                    color: saveStatus === 'error' ? 'var(--accent-pink)' : 'var(--accent-lime)',
                    fontWeight: 'bold',
                    opacity: 0.8
                  }}>
                    {saveStatus === 'saving' ? 'SAVING...' : saveStatus === 'saved' ? 'CONFIG SAVED' : 'SAVE ERROR'}
                  </span>
                )}
              </div>
              <div className="input-field">
                <label>Learning Rate</label>
                <input 
                  type="number" 
                  value={config.lr} 
                  onChange={(e) => updateConfig(prev => ({...prev, lr: parseFloat(e.target.value)}))}
                  step="0.0001"
                />
              </div>
              <div className="input-field">
                <label>Total Epochs</label>
                <input 
                  type="number" 
                  value={config.epochs} 
                  onChange={(e) => updateConfig(prev => ({...prev, epochs: parseInt(e.target.value)}))}
                />
              </div>
              <div className="input-field">
                <label>Batch Size</label>
                <input 
                  type="number" 
                  value={config.batch_size} 
                  onChange={(e) => updateConfig(prev => ({...prev, batch_size: parseInt(e.target.value)}))}
                />
              </div>
            </div>
            
            <div className="config-section" style={{ marginTop: '32px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <h4 className="text-uppercase micro-label" style={{ opacity: 0.7, margin: 0 }}>Data Augmentation</h4>
                <label className="switch-sm" style={{ position: 'relative', display: 'inline-block', width: '32px', height: '16px' }}>
                  <input 
                    type="checkbox" 
                    checked={config.augmentation.enabled}
                    onChange={(e) => updateConfig(prev => ({
                      ...prev, 
                      augmentation: { ...prev.augmentation, enabled: e.target.checked }
                    }))}
                    style={{ opacity: 0, width: 0, height: 0 }}
                  />
                  <span className="slider" style={{ 
                    position: 'absolute', cursor: 'pointer', top: 0, left: 0, right: 0, bottom: 0, 
                    backgroundColor: config.augmentation.enabled ? 'var(--accent-lime)' : '#555', 
                    transition: '.4s', borderRadius: '16px' 
                  }}>
                    <span style={{ 
                      position: 'absolute', height: '12px', width: '12px', left: config.augmentation.enabled ? '18px' : '2px', 
                      bottom: '2px', backgroundColor: 'white', transition: '.4s', borderRadius: '50%' 
                    }}></span>
                  </span>
                </label>
              </div>

              {config.augmentation.enabled && (
                <div className="augmentation-controls" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div className="input-field">
                    <label>Occlusion Prob</label>
                    <input 
                      type="number" 
                      value={config.augmentation.occlusion_prob} 
                      onChange={(e) => updateConfig(prev => ({
                        ...prev, 
                        augmentation: { ...prev.augmentation, occlusion_prob: parseFloat(e.target.value) }
                      }))}
                      step="0.1" min="0" max="1"
                    />
                  </div>
                  <div className="input-field">
                    <label>Flip Prob</label>
                    <input 
                      type="number" 
                      value={config.augmentation.flip_prob} 
                      onChange={(e) => updateConfig(prev => ({
                        ...prev, 
                        augmentation: { ...prev.augmentation, flip_prob: parseFloat(e.target.value) }
                      }))}
                      step="0.1" min="0" max="1"
                    />
                  </div>
                  <div className="input-field">
                    <label>Rotation Range</label>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <input 
                        type="number" 
                        value={config.augmentation.rotation_range[0]} 
                        onChange={(e) => updateConfig(prev => ({
                          ...prev, 
                          augmentation: { 
                            ...prev.augmentation, 
                            rotation_range: [parseInt(e.target.value), prev.augmentation.rotation_range[1]] 
                          }
                        }))}
                        placeholder="Min"
                      />
                      <input 
                        type="number" 
                        value={config.augmentation.rotation_range[1]} 
                        onChange={(e) => updateConfig(prev => ({
                          ...prev, 
                          augmentation: { 
                            ...prev.augmentation, 
                            rotation_range: [prev.augmentation.rotation_range[0], parseInt(e.target.value)] 
                          }
                        }))}
                        placeholder="Max"
                      />
                    </div>
                  </div>
                  <div className="input-field">
                    <label>Scaling Range</label>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <input 
                        type="number" 
                        value={config.augmentation.scaling_range[0]} 
                        onChange={(e) => updateConfig(prev => ({
                          ...prev, 
                          augmentation: { 
                            ...prev.augmentation, 
                            scaling_range: [parseFloat(e.target.value), prev.augmentation.scaling_range[1]] 
                          }
                        }))}
                        step="0.1"
                        placeholder="Min"
                      />
                      <input 
                        type="number" 
                        value={config.augmentation.scaling_range[1]} 
                        onChange={(e) => updateConfig(prev => ({
                          ...prev, 
                          augmentation: { 
                            ...prev.augmentation, 
                            scaling_range: [prev.augmentation.scaling_range[0], parseFloat(e.target.value)] 
                          }
                        }))}
                        step="0.1"
                        placeholder="Max"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="glass card logs-card">
            <h3 className="text-uppercase micro-label" style={{ marginBottom: '12px' }}>Live Logs</h3>
            <div 
              className="logs-viewer" 
              ref={logsContainerRef}
              style={{ maxHeight: '300px', overflowY: 'auto' }}
            >
              {status?.log_history
                .filter(log => !log.includes('|') || !log.includes('%')) // Filter out noisy tqdm lines
                .map((log, i) => {
                  const timeMatch = log.match(/^\[(.*?)\] (.*)/);
                  if (timeMatch) {
                    return (
                      <div className="log-entry" key={i}>
                        <span className="log-time">{timeMatch[1]}</span>
                        <span className="log-text">{timeMatch[2]}</span>
                      </div>
                    );
                  }
                  return (
                    <div className="log-entry" key={i}>
                      {log}
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;
