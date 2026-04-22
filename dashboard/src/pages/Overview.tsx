import React, { useState, useEffect, useCallback } from 'react';
import { getSystemInfo, getTrainingStatus, getModels } from '../services/api';
import { Activity, Cpu, Server, CheckCircle2, PlayCircle } from 'lucide-react';

interface SystemInfo {
  status: string;
  version: string;
  gpu: {
    available: boolean;
    name?: string;
    memory?: {
      free: number;
      total: number;
      used: number;
    };
  };
}

interface TrainingStatus {
  is_running: boolean;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  status_message: string;
}

const Overview: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [latestModel, setLatestModel] = useState<string>('N/A');

  const fetchData = useCallback(async () => {
    try {
      const [sys, train, mods] = await Promise.all([
        getSystemInfo(),
        getTrainingStatus(),
        getModels()
      ]);
      setSystemInfo(sys);
      setTrainingStatus(train);
      if (mods.models && mods.models.length > 0) {
        setLatestModel(mods.models[mods.models.length - 1].name);
      }
    } catch (error) {
      console.error('Failed to fetch overview data:', error);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    
    const fetch = async () => {
      if (isMounted) await fetchData();
    };

    fetch();
    const interval = setInterval(fetch, 5000);
    
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [fetchData]);

  const gpuMemoryUsedPercent = systemInfo?.gpu.memory 
    ? (systemInfo.gpu.memory.used / systemInfo.gpu.memory.total) * 100 
    : 0;

  return (
    <div className="overview-page">
      <h1 className="text-uppercase" style={{ marginBottom: '24px' }}>Dashboard Overview</h1>
      
      <div className="grid-container" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '24px' }}>
        
        {/* Training / Latest Model Card */}
        <div className="glass card" style={{ padding: '24px', borderRadius: '12px', borderLeft: '4px solid var(--accent-lime)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
            <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-lime)' }}>
              {trainingStatus?.is_running ? 'Active Training' : 'Latest Checkpoint'}
            </h3>
            {trainingStatus?.is_running ? (
              <PlayCircle size={20} color="var(--accent-lime)" className="spin-slow" />
            ) : (
              <CheckCircle2 size={20} color="var(--accent-lime)" />
            )}
          </div>
          
          <h2 style={{ margin: '12px 0', fontSize: '1.5rem' }}>
            {trainingStatus?.is_running ? `Epoch ${trainingStatus.current_epoch} / ${trainingStatus.total_epochs}` : latestModel}
          </h2>
          
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Status: {trainingStatus?.status_message || 'Idle'}
          </p>
          
          <div style={{ marginTop: '20px' }}>
            {trainingStatus?.is_running ? (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8rem' }}>
                  <span>Progress</span>
                  <span>{Math.round(trainingStatus.progress * 100)}%</span>
                </div>
                <div style={{ height: '6px', background: 'var(--bg-secondary)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ 
                    width: `${trainingStatus.progress * 100}%`, 
                    height: '100%', 
                    background: 'var(--accent-lime)',
                    transition: 'width 0.5s ease'
                  }}></div>
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity size={16} color="var(--accent-lime)" />
                <span style={{ fontSize: '1.2rem', fontWeight: '700' }}>Ready</span>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>for inference</span>
              </div>
            )}
          </div>
        </div>

        {/* System Status Card */}
        <div className="glass card" style={{ padding: '24px', borderRadius: '12px', borderLeft: '4px solid var(--accent-pink)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
            <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-pink)' }}>System Status</h3>
            {systemInfo?.gpu.available ? (
              <Server size={20} color="var(--accent-pink)" />
            ) : (
              <Cpu size={20} color="var(--text-secondary)" />
            )}
          </div>

          <h2 style={{ margin: '12px 0', fontSize: '1.5rem' }}>
            {systemInfo?.gpu.available ? systemInfo.gpu.name : 'CPU Only'}
          </h2>

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            {systemInfo?.gpu.available && systemInfo.gpu.memory 
              ? `VRAM: ${systemInfo.gpu.memory.used.toFixed(1)} / ${systemInfo.gpu.memory.total.toFixed(1)} GB`
              : `Compute: ${systemInfo?.gpu.available ? 'CUDA Active' : 'Fallback Mode'}`}
          </p>

          <div style={{ marginTop: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8rem' }}>
              <span>{systemInfo?.gpu.available ? 'GPU Load' : 'CPU Load'}</span>
              <span>{systemInfo?.gpu.available && systemInfo.gpu.memory ? `${Math.round(gpuMemoryUsedPercent)}%` : 'Active'}</span>
            </div>
            <div style={{ height: '6px', background: 'var(--bg-secondary)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ 
                width: systemInfo?.gpu.available ? `${gpuMemoryUsedPercent}%` : '100%', 
                height: '100%', 
                background: 'var(--accent-pink)',
                transition: 'width 0.5s ease'
              }}></div>
            </div>
          </div>
        </div>

      </div>
      
      <div style={{ marginTop: '32px', padding: '20px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', border: '1px border-purple' }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: systemInfo?.status === 'online' ? '#4ade80' : '#f87171' }}></span>
          API Backend: {systemInfo?.status || 'Offline'} (v{systemInfo?.version || '0.0.0'})
        </p>
      </div>
    </div>
  );
};

export default Overview;
