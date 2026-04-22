import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Square, 
  Settings, 
  Activity,
  ChevronRight,
  RefreshCw
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
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

interface TrainingStatus {
  is_running: boolean;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  loss_history: number[];
  status_message: string;
}

const Training: React.FC = () => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [config, setConfig] = useState({
    lr: 0.001,
    epochs: 10,
    batch_size: 32,
    remote: false
  });

  const fetchStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/training/status`);
      setStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    try {
      await axios.post(`${API_BASE_URL}/training/start`, {
        training: {
          lr: config.lr,
          epochs: config.epochs,
          batch_size: config.batch_size
        },
        remote: config.remote
      });
      fetchStatus();
    } catch (error) {
      alert('Failed to start training');
    }
  };


  const handleStop = async () => {
    try {
      await axios.post(`${API_BASE_URL}/training/stop`);
      fetchStatus();
    } catch (error) {
      alert('Failed to stop training');
    }
  };

  const chartData = status?.loss_history.map((loss, index) => ({
    epoch: index + 1,
    loss: loss
  })) || [];

  return (
    <div className="training-page">
      <div className="page-header">
        <h1 className="text-uppercase">Training Monitor</h1>
        <p className="text-secondary">Track and control model training sessions.</p>
      </div>

      <div className="training-grid">
        <div className="main-stats">
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
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass card progress-card" style={{ marginTop: '24px' }}>
            <div className="card-header">
              <h3 className="text-uppercase micro-label">Overall Progress</h3>
              <span className="text-secondary">{status?.current_epoch || 0} / {status?.total_epochs || 0} Epochs</span>
            </div>
            <div className="progress-bar-container">
              <div className="progress-bar" style={{ width: `${(status?.progress || 0) * 100}%` }}></div>
            </div>
            <div className="status-footer">
              <span className="text-secondary">{status?.status_message || 'Idle'}</span>
              {status?.is_running && <RefreshCw size={14} className="spin" style={{ marginLeft: '8px' }} />}
            </div>
          </div>
        </div>

        <div className="side-controls">
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
                    onChange={(e) => setConfig({...config, remote: e.target.checked})}
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

              {status?.is_running ? (

                <button className="btn-primary" onClick={handleStop} style={{ width: '100%', background: 'var(--accent-pink)', borderColor: '#8a4d4d' }}>
                  <Square size={18} fill="currentColor" style={{ marginRight: '8px' }} />
                  Stop Training
                </button>
              ) : (
                <button className="btn-lime" onClick={handleStart} style={{ width: '100%' }}>
                  <Play size={18} fill="currentColor" style={{ marginRight: '8px' }} />
                  Start Training
                </button>
              )}
            </div>

            <div className="config-section" style={{ marginTop: '32px' }}>
              <h4 className="text-uppercase micro-label" style={{ marginBottom: '16px', opacity: 0.7 }}>Hyperparameters</h4>
              <div className="input-field">
                <label>Learning Rate</label>
                <input 
                  type="number" 
                  value={config.lr} 
                  onChange={(e) => setConfig({...config, lr: parseFloat(e.target.value)})}
                  step="0.0001"
                />
              </div>
              <div className="input-field">
                <label>Total Epochs</label>
                <input 
                  type="number" 
                  value={config.epochs} 
                  onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                />
              </div>
              <div className="input-field">
                <label>Batch Size</label>
                <input 
                  type="number" 
                  value={config.batch_size} 
                  onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                />
              </div>
            </div>
          </div>
          
          <div className="glass card logs-card" style={{ marginTop: '24px' }}>
            <h3 className="text-uppercase micro-label" style={{ marginBottom: '12px' }}>Live Logs</h3>
            <div className="logs-viewer">
              <div className="log-entry">
                <span className="log-time">[15:04:22]</span> Initializing training pipeline...
              </div>
              <div className="log-entry">
                <span className="log-time">[15:04:23]</span> Model loaded successfully.
              </div>
              {status?.loss_history.map((loss, i) => (
                <div className="log-entry" key={i}>
                  <span className="log-time">[{15 + i}:00:00]</span> Epoch {i+1} - Loss: {loss.toFixed(4)}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;
