import React, { useState, useEffect } from 'react';
import { Server, Shield, CheckCircle, XCircle, RefreshCw, Save, Upload } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

interface GPUConfig {
  type: string;
  tunnel_hostname: string;
  host: string;
  ssh_user: string;
  port: number;
  gpu: string;
  ssh_config_alias: string;
}

const Settings: React.FC = () => {
  const [config, setConfig] = useState<GPUConfig>({
    type: 'cloudflare_tunnel',
    tunnel_hostname: '',
    host: '',
    ssh_user: 'root',
    port: 22,
    gpu: '',
    ssh_config_alias: ''
  });
  const [loading, setLoading] = useState(true);
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<{ success: boolean; stdout: string; stderr: string } | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const content = event.target?.result as string;
        const parsed = JSON.parse(content);
        
        // Map common Kaggle JSON fields to our internal state
        const newConfig = {
          ...config,
          type: parsed.type || (parsed.tunnel_hostname ? 'cloudflare_tunnel' : 'ssh'),
          tunnel_hostname: parsed.tunnel_hostname || '',
          host: parsed.host || parsed.tunnel_hostname || '',
          ssh_user: parsed.ssh_user || 'root',
          port: parsed.port || 22,
          gpu: parsed.gpu || '',
          ssh_config_alias: parsed.ssh_config_alias || ''
        };
        
        setConfig(newConfig);
      } catch (err) {
        alert('Failed to parse JSON file. Ensure it is a valid gpu_connection.json.');
      }
    };
    reader.readAsText(file);
  };


  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/config/gpu`);
      if (response.data && response.data.type) {
        setConfig({ ...config, ...response.data });
      }
    } catch (error) {
      console.error('Failed to fetch GPU config:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaveStatus('saving');
    try {
      await axios.post(`${API_BASE_URL}/config/gpu`, config);
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (error) {
      setSaveStatus('error');
    }
  };

  const handleVerify = async () => {
    setVerifying(true);
    setVerifyResult(null);
    try {
      // Important: Save current config to backend before verifying, 
      // as the verification script reads from the disk file.
      await axios.post(`${API_BASE_URL}/config/gpu`, config);
      
      const response = await axios.post(`${API_BASE_URL}/gpu/verify`);
      setVerifyResult(response.data);
    } catch (error) {
      setVerifyResult({
        success: false,
        stdout: '',
        stderr: 'Failed to trigger verification. Ensure backend is running.'
      });
    } finally {
      setVerifying(false);
    }
  };

  if (loading) return <div style={{ padding: '40px' }}>Loading configuration...</div>;

  return (
    <div className="settings-page">
      <div className="page-header">
        <h1 className="text-uppercase">System Settings</h1>
        <p className="text-secondary">Configure remote GPU connections and environment variables.</p>
      </div>

      <div className="settings-grid" style={{ marginTop: '32px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
        <div className="glass card" style={{ padding: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Server size={24} color="var(--accent-primary)" />
              <h2 className="text-uppercase" style={{ fontSize: '1.2rem', margin: 0 }}>Remote GPU Config</h2>
            </div>
            <label className="glass" style={{ 
              padding: '8px 12px', 
              borderRadius: '8px', 
              fontSize: '0.75rem', 
              cursor: 'pointer', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '6px',
              border: '1px solid var(--border-light)',
              color: 'var(--text-secondary)'
            }}>
              <Upload size={14} />
              IMPORT JSON
              <input type="file" accept=".json" onChange={handleImport} style={{ display: 'none' }} />
            </label>
          </div>

          <div className="form-group" style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Connection Type</label>
            <select 
              className="glass" 
              style={{ width: '100%', padding: '12px', borderRadius: '8px', color: 'white', border: '1px solid var(--border-light)' }}
              value={config.type}
              onChange={(e) => setConfig({ ...config, type: e.target.value })}
            >
              <option value="cloudflare_tunnel">Cloudflare Tunnel (Kaggle)</option>
              <option value="ssh">Direct SSH</option>
            </select>
          </div>

          {config.type === 'cloudflare_tunnel' ? (
            <div className="form-group" style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Tunnel Hostname</label>
              <input 
                type="text" 
                className="glass" 
                style={{ width: '100%', padding: '12px', borderRadius: '8px', color: 'white', border: '1px solid var(--border-light)' }}
                value={config.tunnel_hostname}
                onChange={(e) => setConfig({ ...config, tunnel_hostname: e.target.value })}
                placeholder="e.g. abc-def.trycloudflare.com"
              />
            </div>
          ) : (
            <div className="form-group" style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Host Address</label>
              <input 
                type="text" 
                className="glass" 
                style={{ width: '100%', padding: '12px', borderRadius: '8px', color: 'white', border: '1px solid var(--border-light)' }}
                value={config.host}
                onChange={(e) => setConfig({ ...config, host: e.target.value })}
                placeholder="e.g. 1.2.3.4"
              />
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div className="form-group" style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>SSH User</label>
              <input 
                type="text" 
                className="glass" 
                style={{ width: '100%', padding: '12px', borderRadius: '8px', color: 'white', border: '1px solid var(--border-light)' }}
                value={config.ssh_user}
                onChange={(e) => setConfig({ ...config, ssh_user: e.target.value })}
              />
            </div>
            <div className="form-group" style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Port</label>
              <input 
                type="number" 
                className="glass" 
                style={{ width: '100%', padding: '12px', borderRadius: '8px', color: 'white', border: '1px solid var(--border-light)' }}
                value={config.port}
                onChange={(e) => setConfig({ ...config, port: parseInt(e.target.value) })}
              />
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>GPU Model (Display only)</label>
            <input 
              type="text" 
              className="glass" 
              style={{ width: '100%', padding: '12px', borderRadius: '8px', color: 'white', border: '1px solid var(--border-light)' }}
              value={config.gpu}
              readOnly
              placeholder="Detecting..."
            />
          </div>

          <div style={{ display: 'flex', gap: '16px', marginTop: '12px' }}>
            <button 
              className="btn-primary" 
              style={{ flex: 1, padding: '12px' }}
              onClick={handleSave}
              disabled={saveStatus === 'saving'}
            >
              <Save size={18} style={{ marginRight: '8px' }} />
              {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'success' ? 'Saved!' : 'Save Config'}
            </button>
            <button 
              className="glass" 
              style={{ flex: 1, padding: '12px', border: '1px solid var(--border-light)', borderRadius: '8px', color: 'white', cursor: 'pointer' }}
              onClick={handleVerify}
              disabled={verifying}
            >
              <RefreshCw size={18} style={{ marginRight: '8px', animation: verifying ? 'spin 1s linear infinite' : 'none' }} />
              {verifying ? 'Verifying...' : 'Verify Connection'}
            </button>
          </div>
        </div>

        <div className="glass card" style={{ padding: '32px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
            <Shield size={24} color="var(--accent-pink)" />
            <h2 className="text-uppercase" style={{ fontSize: '1.2rem', margin: 0 }}>Connection Status</h2>
          </div>

          {verifyResult ? (
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', color: verifyResult.success ? 'var(--accent-primary)' : 'var(--accent-pink)' }}>
                {verifyResult.success ? <CheckCircle size={20} /> : <XCircle size={20} />}
                <span style={{ fontWeight: 'bold' }}>{verifyResult.success ? 'CONNECTION SUCCESSFUL' : 'CONNECTION FAILED'}</span>
              </div>
              
              <div className="logs-panel glass" style={{ height: '300px', padding: '16px', fontSize: '0.75rem', overflowY: 'auto', background: 'rgba(0,0,0,0.3)' }}>
                <pre style={{ margin: 0, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                  {verifyResult.stdout}
                  {verifyResult.stderr && <span style={{ color: 'var(--accent-pink)' }}>{verifyResult.stderr}</span>}
                </pre>
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', color: 'var(--text-secondary)' }}>
              <RefreshCw size={48} style={{ marginBottom: '16px', opacity: 0.2 }} />
              <p>Run verification to check remote GPU availability.</p>
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default Settings;
