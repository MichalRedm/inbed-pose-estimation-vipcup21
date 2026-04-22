import React, { useState, useEffect } from 'react';
import { Box, FileText, Download, Trash2, CheckCircle } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

interface ModelCheckpoint {
  name: string;
  path: string;
  size_mb: number;
}

const Models: React.FC = () => {
  const [models, setModels] = useState<ModelCheckpoint[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models`);
      setModels(response.data.models);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  return (
    <div className="models-page">
      <div className="page-header">
        <h1 className="text-uppercase">Model Management</h1>
        <p className="text-secondary">View and manage trained model checkpoints.</p>
      </div>

      <div className="models-grid" style={{ marginTop: '32px', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '24px' }}>
        {loading ? (
          <p>Loading models...</p>
        ) : models.length > 0 ? (
          models.map((model, i) => (
            <div key={i} className="glass card model-card" style={{ padding: '24px', borderRadius: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <Box size={24} color="var(--accent-primary)" />
                <h3 style={{ fontSize: '1.1rem' }}>{model.name}</h3>
              </div>
              <div className="model-meta" style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Size:</span>
                  <span style={{ color: 'var(--text-primary)' }}>{model.size_mb.toFixed(1)} MB</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Path:</span>
                  <span style={{ color: 'var(--text-primary)', fontSize: '0.75rem' }}>{model.path}</span>
                </div>
              </div>
              <div className="model-actions" style={{ marginTop: '24px', display: 'flex', gap: '12px' }}>
                <button className="btn-primary" style={{ flex: 1, padding: '8px', fontSize: '0.75rem' }}>
                  <CheckCircle size={14} style={{ marginRight: '6px' }} />
                  Activate
                </button>
                <button className="icon-btn glass" style={{ padding: '8px', borderRadius: '8px' }}>
                  <Download size={16} />
                </button>
                <button className="icon-btn glass" style={{ padding: '8px', borderRadius: '8px', color: 'var(--accent-pink)' }}>
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          ))
        ) : (
          <div className="glass card" style={{ padding: '40px', gridColumn: '1 / -1', textAlign: 'center' }}>
            <p className="text-secondary">No model checkpoints found in `models/checkpoints/`.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Models;
