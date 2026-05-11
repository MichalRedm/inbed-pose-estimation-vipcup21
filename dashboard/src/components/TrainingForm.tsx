import React, { useState, useEffect } from 'react';
import { Play, Server, Activity } from 'lucide-react';
import { getTrainingConfig, saveTrainingConfig, startTraining } from '../services/api';

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

interface TrainingFormProps {
  onStarted: () => void;
}

const TrainingForm: React.FC<TrainingFormProps> = ({ onStarted }) => {
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

  useEffect(() => {
    const initialize = async () => {
      try {
        const savedConfig = await getTrainingConfig();
        if (savedConfig) {
          setConfig(prev => ({
            ...prev,
            ...savedConfig,
            anatomical: savedConfig.lambda_anatomical > 0
          }));
        }
      } catch (error) {
        console.error('Failed to load training config:', error);
      }
    };
    initialize();
  }, []);


  const handleStart = async () => {
    try {
      await saveTrainingConfig(config as any);
      await startTraining(config as any);
      onStarted();
    } catch (error) {
      console.error('Failed to start training:', error);
      alert('Failed to start training');
    }
  };

  return (
    <div className="training-form glass card" style={{ maxWidth: '600px', margin: '0 auto' }}>
      <div className="card-header">
        <h2 className="text-uppercase">New Training Session</h2>
        <p className="text-secondary">Configure hyperparameters and launch a new experiment loop.</p>
      </div>

      <div className="flex-column" style={{ gap: '20px' }}>
        <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
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
        </div>

        <div className="input-field">
          <label>Batch Size</label>
          <input 
            type="number" 
            value={config.batch_size} 
            onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
          />
        </div>

        <div className="config-section" style={{ paddingTop: '20px', borderTop: '1px solid var(--border-purple)' }}>
           <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Server size={18} color="var(--accent-primary)" />
              <span style={{ fontWeight: 600 }}>Remote Training (Kaggle/SSH)</span>
            </div>
            <label className="switch">
              <input 
                type="checkbox" 
                checked={config.remote}
                onChange={(e) => setConfig({...config, remote: e.target.checked})}
              />
              <span className="slider"></span>
            </label>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} color="var(--accent-pink)" />
              <span style={{ fontWeight: 600 }}>UDA (Domain Adaptation)</span>
            </div>
            <label className="switch">
              <input 
                type="checkbox" 
                checked={config.uda}
                onChange={(e) => setConfig({...config, uda: e.target.checked})}
              />
              <span className="slider"></span>
            </label>
          </div>

          {config.uda && (
             <div className="input-field" style={{ marginLeft: '24px' }}>
              <label>Lambda Adversarial</label>
              <input 
                type="number" 
                value={config.lambda_adv} 
                onChange={(e) => setConfig({...config, lambda_adv: parseFloat(e.target.value)})}
                step="0.05"
              />
            </div>
          )}
        </div>

        <button className="btn-lime" onClick={handleStart} style={{ width: '100%', marginTop: '20px' }}>
          <Play size={18} fill="currentColor" />
          Launch Run
        </button>
      </div>
    </div>
  );
};

export default TrainingForm;
